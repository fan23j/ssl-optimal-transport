import torch
import matplotlib.pyplot as plt

from tqdm import tqdm

from ..base_trainer import BaseTrainer


class InferenceSinkhornTrainer(BaseTrainer):
    def __init__(self, cfg, model, optimizer, lr_scheduler, train_dataset, val_dataset):
        super(InferenceSinkhornTrainer, self).__init__(
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
        prediction = torch.argsort(sim_matrix, dim=-1, descending=True)
        total_correct_1 = torch.sum(
                (prediction[:, 0:1] == targets_whole.unsqueeze(dim=-1)).any(dim=-1).float()
        ).item()
        total_correct_5 = torch.sum(
            (prediction[:, 0:5] == targets_whole.unsqueeze(dim=-1)).any(dim=-1).float()
        ).item()
        print("Top1: {:.2f}%".format(total_correct_1 / total_num * 100))
        print("Top5: {:.2f}%".format(total_correct_5 / total_num * 100))

    def val(self, epoch, test_data_loader):
        return self.train(epoch, test_data_loader, is_train=False)