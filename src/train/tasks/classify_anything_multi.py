import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import average_precision_score

from ..base_trainer import BaseTrainer

# Disable warnings
warnings.filterwarnings("ignore", category=UserWarning)


class ClassifyAnythingMultiTrainer(BaseTrainer):
    def __init__(self, cfg, model, optimizer, lr_scheduler, dataset):
        super(ClassifyAnythingMultiTrainer, self).__init__(
            cfg, model, optimizer, lr_scheduler
        )
        print("Loading pre-computed word vectors...")
        self.label_vectors = torch.load(cfg.MODEL.LABEL_VECTORS)
        self.model = model
        with torch.no_grad():
            self.label_vectors = (
                torch.tensor(self.label_vectors).float().cuda(non_blocking=True)
            )

    def train(self, epoch, data_loader, is_train=True):
        self.model.train() if is_train else self.model.eval()
        data_bar = tqdm(data_loader)
        average_loss_states = {}
        # collect all targets and predictions
        all_targets = []
        all_preds = []

        with torch.enable_grad() if is_train else torch.no_grad():
            for it, batch in enumerate(data_bar):
                data = batch["out_1"]
                target = batch["target"]
                data, target = (
                    data.cuda(non_blocking=True),
                    target.cuda(non_blocking=True),
                )

                features = self.model(data)

                projected_label_vectors = self.model.module.labels_proj_head(
                    self.label_vectors
                )

                loss, loss_states, cosim_softmax = self.loss(
                    features=features,
                    labels_vector=projected_label_vectors,
                    targets=target,
                    model=self.model.module,
                )

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

                # apply sigmoid to cosim_softmax to get predictions between 0 and 1
                preds = torch.sigmoid(cosim_softmax)

                all_preds.append(preds.cpu())
                all_targets.append(target.cpu())

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

                data_bar.set_description(
                    "{} Epoch: [{}/{}] {} mAP: {:.2f}%".format(
                        "Train" if is_train else "Test",
                        epoch,
                        self.cfg.TRAIN.EPOCHS,
                        loss_state_str,
                        mAP * 100,
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
