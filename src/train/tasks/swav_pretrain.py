import torch
import os
from tqdm import tqdm
import torch.nn.functional as F
import apex


from ..base_trainer import BaseTrainer


class SwAVPreTrainer(BaseTrainer):
    def __init__(self, cfg, model, optimizer, lr_scheduler):
        super(SwAVPreTrainer, self).__init__(cfg, model, optimizer, lr_scheduler)
        self.rank = cfg.TRAIN.LOCAL_RANK if cfg.TRAIN.DISTRIBUTE else 1
        # build the queue
        self.queue = None
        queue_path = os.path.join(
            cfg.OUTPUT_DIR,
            cfg.EXP_ID,
            "queue" + str(self.rank) + ".pth",
        )
        if os.path.isfile(queue_path):
            self.queue = torch.load(queue_path)["queue"]
        # the queue needs to be divisible by the batch size
        assert cfg.TRAIN.QUEUE_LENGTH % (cfg.TRAIN.BATCH_SIZE * self.rank) == 0

    def train(self, epoch, data_loader):
        # optionally starts a queue
        if (
            self.cfg.TRAIN.QUEUE_LENGTH > 0
            and epoch >= self.cfg.TRAIN.EPOCH_QUEUE_START
            and self.queue is None
        ):
            self.queue = torch.zeros(
                len(self.cfg.TRAIN.CROPS_FOR_ASSIGN),
                self.cfg.TRAIN.QUEUE_LENGTH // self.rank,
                self.cfg.MODEL.SWAV_FEAT_DIM,
            ).cuda()

        self.model.train()

        train_bar = tqdm(data_loader)
        average_loss_states = {}

        for it, batch in enumerate(train_bar):
            iteration = epoch * len(data_loader) + it

            # normalize the prototypes
            with torch.no_grad():
                w = self.model.module.prototypes.weight.data.clone()
                w = F.normalize(w, dim=1, p=2)
                self.model.module.prototypes.weight.copy_(w)

            # ============ multi-res forward passes ... ============
            embedding, output = self.model(batch)
            embedding = embedding.detach()

            # ============ swav loss ... ============
            loss, loss_states, _ = self.loss(output, self.queue, self.model, embedding)

            # ============ backward and optim step ... ============
            self.optimizer.zero_grad()
            if self.cfg.TRAIN.USE_MIXED_PRECISION:
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            # cancel gradients for the prototypes
            if iteration < self.cfg.TRAIN.FREEZE_PROTOTYPES_NITERS:
                for name, p in self.model.named_parameters():
                    if "prototypes" in name:
                        p.grad = None
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
