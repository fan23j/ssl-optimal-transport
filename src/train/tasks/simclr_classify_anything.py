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


class SimCLRClassifyAnythingTrainer(BaseTrainer):
    def __init__(self, cfg, model, optimizer, lr_scheduler):
        super(SimCLRClassifyAnythingTrainer, self).__init__(
            cfg, model, optimizer, lr_scheduler
        )
        print("Loading pre-computed word vectors...")
        self.label_vectors = torch.load(cfg.MODEL.LABEL_VECTORS)
        self.negative_vectors = torch.load(cfg.MODEL.NEGATIVE_VECTORS)

        with torch.no_grad():
            self.label_vectors = (
                torch.tensor(self.label_vectors).float().cuda(non_blocking=True)
            )
            self.negative_vectors = (
                torch.tensor(self.negative_vectors).float().cuda(non_blocking=True)
            )

    def train(self, epoch, data_loader, is_train=True):
        self.model.train() if is_train else self.model.eval()
        mAP_all, total_num, data_bar = (
            0.0,
            0,
            tqdm(data_loader),
        )
        average_loss_states = {}

        with torch.enable_grad() if is_train else torch.no_grad():
            for it, batch in enumerate(data_bar):
                data = batch["out_1"]
                target = batch["target"]
                data, target = (
                    data.cuda(non_blocking=True),
                    target.cuda(non_blocking=True),
                )
                features = self.model(data)

                loss, loss_states, cosim_softmax = self.loss(
                    features=features,
                    labels_vector=self.label_vectors,
                    negative_vectors=self.negative_vectors,
                    targets=target,
                )

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_num += data.size(0)
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

                average_precisions = []
                for class_idx in range(target.shape[1]):
                    class_preds = preds[:, class_idx].cpu().detach().numpy()
                    class_targets = target[:, class_idx].cpu().detach().numpy()
                    try:
                        average_precision = average_precision_score(
                            class_targets, class_preds
                        )
                        average_precisions.append(average_precision)
                    except UndefinedMetricWarning:
                        pass  # Ignore this specific warning
                mAP = np.mean(average_precisions)
                mAP_all += mAP
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

        average_loss_states["mAP"] = mAP_all / total_num * 100
        return average_loss_states

    def val(self, epoch, test_data_loader):
        return self.train(epoch, test_data_loader, is_train=False)
