import torch
import matplotlib.pyplot as plt

from tqdm import tqdm

import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import average_precision_score
import numpy as np
from ..base_trainer import BaseTrainer

# Disable warnings
warnings.filterwarnings("ignore", category=UserWarning)


class InferenceMultilabelTrainer(BaseTrainer):
    def __init__(self, cfg, model, optimizer, lr_scheduler, train_dataset, val_dataset):
        super(InferenceMultilabelTrainer, self).__init__(
            cfg, model, optimizer, lr_scheduler, train_dataset, val_dataset
        )
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.dataset = train_dataset
        self.cfg = cfg

    def train(self, epoch, data_loader, is_train=True):
        self.model.train() if is_train else self.model.eval()
        total_correct_1, total_correct_5, total_num, data_bar = (
            0.0,
            0.0,
            0,
            tqdm(data_loader),
        )
        average_loss_states = {}
        figure = None
        sim_matrices = []
        targets_list = []

        with torch.enable_grad() if is_train else torch.no_grad():
            for it, batch in enumerate(data_bar):
                data = batch["out_1"]
                target = batch["target"]
                data, target = (
                    data.cuda(non_blocking=True),
                    target.cuda(non_blocking=True),
                )
                features = self.model(data)

                if self.cfg.TRAIN.USE_MULTICLASS:
                    text_features = self.model.module.backbone_model.encode_text(
                        self.dataset.multiclass_text_inputs.to("cuda")
                    )
                else:
                    text_features = self.model.module.backbone_model.encode_text(
                        self.dataset.multilabel_text_inputs.to("cuda")
                    )
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                _, _, sim_matrix = self.loss(
                    features=features,
                    text_features=text_features,
                    targets=target,
                    dataset=self.train_dataset,
                )
                sim_matrices.append(sim_matrix)
                targets_list.append(target)
                
                total_num += data.size(0)


        sim_matrix_whole = torch.cat(sim_matrices, dim=0)
        targets_whole = torch.cat(targets_list, dim=0)
        _, _, sim_matrix = self.loss(
                    features=features,
                    text_features=text_features,
                    targets=target,
                    dataset=self.train_dataset,
                    sim_matrix_whole=sim_matrix_whole,
                    total_num=total_num,
        )
        
        multilabel_preds = torch.sigmoid(sim_matrix)
        cur_preds = multilabel_preds.cpu()
        cur_targets = torch.cat(targets_list, dim=0)

        average_precisions = []
        for class_idx in range(cur_targets.shape[1]):
            #import pudb;
            class_preds = cur_preds[:,:,0][:, class_idx].detach().numpy()
            # max_vals, _ = torch.min(cur_preds, dim=2)
            # class_preds = max_vals[:, class_idx].detach().numpy()
            class_targets = cur_targets[:, class_idx].cpu().detach().numpy()

            try:
                average_precision = average_precision_score(
                    class_targets, class_preds
                )
                average_precisions.append(average_precision)
            except UndefinedMetricWarning:
                pass  # Ignore this specific warning
        mAP = np.mean(average_precisions)
        print("mAP:", mAP * 100)

    def val(self, epoch, test_data_loader):
        return self.train(epoch, test_data_loader, is_train=False)