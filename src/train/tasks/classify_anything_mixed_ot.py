import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import average_precision_score

from ..base_trainer import BaseTrainer

# Disable warnings
warnings.filterwarnings("ignore", category=UserWarning)


class ClassifyAnythingMixedOtTrainer(BaseTrainer):
    def __init__(self, cfg, model, optimizer, lr_scheduler, train_dataset, val_dataset):
        super(ClassifyAnythingMixedOtTrainer, self).__init__(
            cfg, model, optimizer, lr_scheduler, train_dataset, val_dataset
        )
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.dataset = train_dataset

    def train(self, epoch, data_loader, is_train=True):
        self.model.train() if is_train else self.model.eval()
        data_bar = tqdm(data_loader)
        self.dataset = self.train_dataset if is_train else self.val_dataset
        average_loss_states = {}
        # collect all targets and predictions
        all_targets = []
        all_preds = []
        total_correct_1 = 0
        total_correct_5 = 0
        multiclass_sample_count = 0
        with torch.enable_grad() if is_train else torch.no_grad():
            self.dataset.on_epoch_start()
            for it, (batch_data, dataset_indices) in enumerate(data_bar):
                data, targets = batch_data["out_1"], batch_data["target"]
                
                data = data.cuda(non_blocking=True)

                features = self.model(data)

                # multilabel
                multilabel_text_features = self.model.module.backbone_model.encode_text(
                    self.dataset.multilabel_text_inputs.to("cuda")
                )
                multilabel_text_features = (
                    multilabel_text_features
                    / multilabel_text_features.norm(dim=-1, keepdim=True)
                )

                # multiclass
                multiclass_text_features = self.model.module.backbone_model.encode_text(
                    self.dataset.multiclass_text_inputs.to("cuda")
                )
                multiclass_text_features = (
                    multiclass_text_features
                    / multiclass_text_features.norm(dim=-1, keepdim=True)
                )

                loss, loss_states, payload = self.loss(
                    features=features,
                    multilabel_text_features=multilabel_text_features,
                    multiclass_text_features=multiclass_text_features,
                    targets=targets,
                    dataset_indices=dataset_indices,
                    dataset=self.dataset,
                )

                (
                    multilabel_cosim,
                    multiclass_cosim,
                    multilabel_targets,
                    multiclass_targets,
                ) = payload

                # Separating out targets based on dataset_indices
                multilabel_targets = multilabel_targets[dataset_indices == 0]
                multiclass_targets = multiclass_targets[dataset_indices == 1]

                # Converting one-hot encoded multiclass targets to class indices
                _, multiclass_targets_indices = multiclass_targets.max(dim=1)
                multiclass_sample_count += len(multiclass_targets_indices)
                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Accumulate average_loss_states
                for k, v in loss_states.items():
                    if k not in average_loss_states:
                        average_loss_states[k] = 0.0
                    average_loss_states[k] += v.item()

                # Create a string of loss states
                loss_state_str = ", ".join(
                    f"{k}: {v:.4f}" for k, v in loss_states.items()
                )

                # multilabel: mAP calculation using separated targets and predictions
                multilabel_preds = torch.sigmoid(multilabel_cosim)
                all_preds.append(multilabel_preds.cpu())
                all_targets.append(multilabel_targets.cpu())

                # calculate mAP for current collected predictions and targets
                cur_preds = torch.cat(all_preds, dim=0)
                cur_targets = torch.cat(all_targets, dim=0)

                average_precisions = []
                for class_idx in range(cur_targets.shape[1]):
                    class_preds = cur_preds[:, class_idx].detach().numpy()
                    class_targets = cur_targets[:, class_idx].detach().numpy()

                    try:
                        average_precision = average_precision_score(
                            class_targets, class_preds
                        )
                        average_precisions.append(average_precision)
                    except UndefinedMetricWarning:
                        pass  # Ignore this specific warning
                mAP = np.mean(average_precisions)

                # multiclass: Top-k accuracy calculation
                multiclass_cosim_softmax = F.softmax(multiclass_cosim, dim=1)
                prediction = torch.argsort(
                    multiclass_cosim_softmax, dim=-1, descending=True
                ).cpu()
                total_correct_1 += torch.sum(
                    (prediction[:, 0:1] == multiclass_targets_indices.unsqueeze(dim=-1))
                    .any(dim=-1)
                    .float()
                ).item()
                total_correct_5 += torch.sum(
                    (prediction[:, 0:5] == multiclass_targets_indices.unsqueeze(dim=-1))
                    .any(dim=-1)
                    .float()
                ).item()

                top1_acc = (total_correct_1 / multiclass_sample_count) * 100
                top5_acc = (total_correct_5 / multiclass_sample_count) * 100

                data_bar.set_description(
                    "{} Epoch: [{}/{}] {} mAP: {:.2f}%, Top-1: {:.2f}%, Top-5: {:.2f}%".format(
                        "Train" if is_train else "Test",
                        epoch,
                        self.cfg.TRAIN.EPOCHS,
                        loss_state_str,
                        mAP * 100,
                        top1_acc,
                        top5_acc,
                    )
                )

        self.lr_scheduler.step()
        # Average the accumulated loss_states
        for k in average_loss_states:
            average_loss_states[k] /= len(data_loader)

        average_loss_states["mAP"] = mAP * 100
        average_loss_states["Top_1"] = top1_acc
        average_loss_states["Top_5"] = top5_acc
        average_loss_states["metric"] = average_loss_states["mAP"]
        return average_loss_states

    def val(self, epoch, test_data_loader):
        return self.train(epoch, test_data_loader, is_train=False)
