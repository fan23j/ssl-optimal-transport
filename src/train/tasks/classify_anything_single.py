import torch
import matplotlib.pyplot as plt

from tqdm import tqdm

from ..base_trainer import BaseTrainer


class ClassifyAnythingSingleTrainer(BaseTrainer):
    def __init__(self, cfg, model, optimizer, lr_scheduler):
        super(ClassifyAnythingSingleTrainer, self).__init__(
            cfg, model, optimizer, lr_scheduler
        )
        print("Loading pre-computed word vectors...")
        self.label_vectors = torch.load(cfg.MODEL.LABEL_VECTORS)

        with torch.no_grad():
            self.label_vectors = (
                torch.tensor(self.label_vectors).float().cuda(non_blocking=True)
            )

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
                    features, self.label_vectors, target
                )

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                elif it == 0:
                    # save cosim_softmax for first batch in val
                    plt.imshow(cosim_softmax.cpu().detach().numpy(), cmap="viridis")
                    plt.colorbar()
                    figure = plt

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

                prediction = torch.argsort(cosim_softmax, dim=-1, descending=True)
                total_correct_1 += torch.sum(
                    (prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()
                ).item()
                total_correct_5 += torch.sum(
                    (prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()
                ).item()

                data_bar.set_description(
                    "{} Epoch: [{}/{}] {} ACC@1: {:.2f}% ACC@5: {:.2f}%".format(
                        "Train" if is_train else "Test",
                        epoch,
                        self.cfg.TRAIN.EPOCHS,
                        loss_state_str,
                        total_correct_1 / total_num * 100,
                        total_correct_5 / total_num * 100,
                    )
                )
        self.lr_scheduler.step()
        # Average the accumulated loss_states
        for k in average_loss_states:
            average_loss_states[k] /= len(data_loader)

        average_loss_states["ACC@1"] = total_correct_1 / total_num * 100
        average_loss_states["ACC@5"] = total_correct_5 / total_num * 100
        average_loss_states["metric"] = average_loss_states["ACC@1"]
        if not is_train:
            average_loss_states["figure"] = figure
        return average_loss_states

    def val(self, epoch, test_data_loader):
        return self.train(epoch, test_data_loader, is_train=False)