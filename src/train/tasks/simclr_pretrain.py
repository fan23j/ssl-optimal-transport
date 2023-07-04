import torch
from tqdm import tqdm

from ..base_trainer import BaseTrainer


class SimCLRPreTrainer(BaseTrainer):
    def __init__(self, cfg, model, optimizer, lr_scheduler):
        super(SimCLRPreTrainer, self).__init__(cfg, model, optimizer, lr_scheduler)

    def train(self, epoch, data_loader):
        self.model.train()
        train_bar = tqdm(data_loader)
        average_loss_states = {}

        for batch in train_bar:
            pos_1 = batch["out_1"]
            pos_2 = batch["out_2"]
            pos_1, pos_2 = pos_1.to(
                device=torch.device("cuda:%d" % self.local_rank), non_blocking=True
            ), pos_2.to(
                device=torch.device("cuda:%d" % self.local_rank), non_blocking=True
            )
            feature_1, out_1 = self.model(pos_1)
            feature_2, out_2 = self.model(pos_2)

            loss, loss_states = self.loss(out_1, out_2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Accumulate average_loss_states
            for k, v in loss_states.items():
                if k not in average_loss_states:
                    average_loss_states[k] = 0.0
                average_loss_states[k] += v.item()

            # Create a string of loss states
            loss_state_str = ", ".join(f"{k}: {v:.4f}" for k, v in loss_states.items())
            train_bar.set_description(
                "Train Epoch: [{}/{}] {}".format(
                    epoch, self.cfg.TRAIN.EPOCHS, loss_state_str
                )
            )
        self.lr_scheduler.step()
        # Average the accumulated loss_states
        for k in average_loss_states:
            average_loss_states[k] /= len(data_loader)

        return average_loss_states

    # test for one epoch, use weighted knn to find the most similar images' label to assign the test image
    def val(self, epoch, test_data_loader):
        self.model.eval()
        total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
        with torch.no_grad():
            # generate feature bank
            for data, _, target in tqdm(test_data_loader, desc="Feature extracting"):
                feature, out = self.model(data.cuda(non_blocking=True))
                feature_bank.append(feature)
            # [D, N]
            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            # [N]
            feature_labels = torch.tensor(
                test_data_loader.dataset.targets, device=feature_bank.device
            )
            # loop test data to predict the label by weighted knn search
            test_bar = tqdm(test_data_loader)
            for data, _, target in test_bar:
                data, target = data.cuda(non_blocking=True), target.cuda(
                    non_blocking=True
                )
                feature, out = self.model(data)

                total_num += data.size(0)
                # compute cos similarity between each feature vector and feature bank ---> [B, N]
                sim_matrix = torch.mm(feature, feature_bank)
                # [B, K]
                sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
                # [B, K]
                sim_labels = torch.gather(
                    feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices
                )
                sim_weight = (sim_weight / self.temperature).exp()

                # counts for each class
                one_hot_label = torch.zeros(
                    data.size(0) * k, c, device=sim_labels.device
                )
                # [B*K, C]
                one_hot_label = one_hot_label.scatter(
                    dim=-1, index=sim_labels.view(-1, 1), value=1.0
                )
                # weighted score ---> [B, C]
                pred_scores = torch.sum(
                    one_hot_label.view(data.size(0), -1, c)
                    * sim_weight.unsqueeze(dim=-1),
                    dim=1,
                )

                pred_labels = pred_scores.argsort(dim=-1, descending=True)
                total_top1 += torch.sum(
                    (pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()
                ).item()
                total_top5 += torch.sum(
                    (pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()
                ).item()
                test_bar.set_description(
                    "Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%".format(
                        epoch,
                        self.cfg.TRAIN.EPOCHS,
                        total_top1 / total_num * 100,
                        total_top5 / total_num * 100,
                    )
                )

        return total_top1 / total_num * 100, total_top5 / total_num * 100
