import torch
from tqdm import tqdm

from ..base_trainer import BaseTrainer


class MAEPreTrainer(BaseTrainer):
    def __init__(self, cfg, model, optimizer):
        super(MAEPreTrainer, self).__init__(cfg, model, optimizer)

    def train(self, epoch, data_loader):
        self.model.train()
        train_bar = tqdm(data_loader)
        average_loss_states = {}

        for batch in train_bar:
            imgs = batch["out_1"]

            pred = self.model(imgs)

            loss, loss_states = self.loss(imgs, pred, self.model.mask)

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

        # Average the accumulated loss_states
        for k in average_loss_states:
            average_loss_states[k] /= len(data_loader)

        return average_loss_states
