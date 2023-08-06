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


class ClassifyAnythingMixedTrainer(BaseTrainer):
    def __init__(self, cfg, model, optimizer, lr_scheduler, train_dataset, val_dataset):
        super(ClassifyAnythingMixedTrainer, self).__init__(
            cfg, model, optimizer, lr_scheduler, train_dataset, val_dataset
        )
        print("Loading pre-computed word vectors...")
        self.label_vectors = torch.load(cfg.MODEL.LABEL_VECTORS)

        # Stack the label vectors from the dictionary
        self.label_vectors = {k: torch.tensor(v) for k, v in self.label_vectors.items()}
        self.label_vectors = torch.stack(list(self.label_vectors.values())).squeeze(1)

        with torch.no_grad():
            self.label_vectors = self.label_vectors.cuda()

    def train(self, epoch, data_loader, is_train=True):
        self.model.train() if is_train else self.model.eval()
        data_bar = tqdm(data_loader)

        average_loss_states = {}
        # collect all targets and predictions
        all_targets = []
        all_preds = []
        total_correct_1 = 0
        total_correct_5 = 0
        with torch.enable_grad() if is_train else torch.no_grad():
            self.train_dataset.on_epoch_start() if is_train else self.val_dataset.on_epoch_start()
            for it, (batch_data, dataset_indices) in enumerate(data_bar):
                data, targets = batch_data["out_1"], batch_data["target"]

                data = data.cuda(non_blocking=True)

                features = self.model(data)

                loss, loss_states, cosim_matrices = self.loss(
                    features=features,
                    label_vectors=self.label_vectors,
                    targets=targets,
                    dataset_indices=dataset_indices,
                    model=self.model.module,
                )

                coco_cosim, cifar_cosim = cosim_matrices

                # Separating out COCO and CIFAR targets based on dataset_indices
                coco_targets = targets[dataset_indices == 0]
                cifar_targets = targets[dataset_indices == 1]

                # Converting one-hot encoded CIFAR targets to class indices
                _, cifar_targets_indices = cifar_targets.max(dim=1)

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

                # COCO: mAP calculation using separated targets and predictions
                coco_preds = torch.sigmoid(coco_cosim)
                all_preds.append(coco_preds.cpu())
                all_targets.append(coco_targets.cpu())

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

                # CIFAR: Top-k accuracy calculation
                cifar_cosim_softmax = F.softmax(cifar_cosim, dim=1)
                prediction = torch.argsort(
                    cifar_cosim_softmax, dim=-1, descending=True
                ).cpu()
                total_correct_1 += torch.sum(
                    (prediction[:, 0:1] == cifar_targets_indices.unsqueeze(dim=-1))
                    .any(dim=-1)
                    .float()
                ).item()
                total_correct_5 += torch.sum(
                    (prediction[:, 0:5] == cifar_targets_indices.unsqueeze(dim=-1))
                    .any(dim=-1)
                    .float()
                ).item()

                top1_acc = (total_correct_1 / len(data_loader.dataset)) * 100
                top5_acc = (total_correct_5 / len(data_loader.dataset)) * 100

                data_bar.set_description(
                    "{} Epoch: [{}/{}] {} COCO mAP: {:.2f}%, CIFAR Top-1: {:.2f}%, Top-5: {:.2f}%".format(
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
        average_loss_states["metric"] = average_loss_states["mAP"]
        return average_loss_states

    def val(self, epoch, test_data_loader):
        return self.train(epoch, test_data_loader, is_train=False)
