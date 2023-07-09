import torch
import copy
from tqdm import tqdm
from einops import rearrange
from model.model import create_model, load_model

from ..base_trainer import BaseTrainer


class MAEPreTrainer(BaseTrainer):
    def __init__(self, cfg, model, optimizer, lr_scheduler):
        super(MAEPreTrainer, self).__init__(cfg, model, optimizer, lr_scheduler)
        # 4096 batch size to achieve SOTA in original paper
        self.steps_per_update = 4096 // cfg.TRAIN.BATCH_SIZE
        self.step_count = 0
        self.supervision = None
        # ============ SimCLR supervision ============
        if self.cfg.MODEL.SUPERVISION != "":
            supervision_cfg = copy.deepcopy(cfg)

            # Update supervision_cfg with new cfg file
            supervision_cfg.merge_from_file(cfg.MODEL.SUPERVISION)

            with torch.no_grad():
                print("Loading supervision model...")
                self.supervision = create_model(
                    supervision_cfg.MODEL.NAME, supervision_cfg
                )
            self.supervision = load_model(supervision_cfg, self.supervision)[0]

    def train(self, epoch, data_loader):
        self.model.train()
        train_bar = tqdm(data_loader)
        average_loss_states = {}
        self.optimizer.zero_grad()
        for batch in train_bar:
            self.step_count += 1
            imgs = batch["out_1"]

            pred, mask, mae_features = self.model(imgs)

            if self.supervision:
                # if supervised model cfg is provided, use task supervision
                _, simclr_features = self.supervision(imgs)
                simclr_features = simclr_features.cuda(non_blocking=True)
                loss, loss_states, _ = self.loss(
                    imgs=imgs,
                    pred=pred,
                    mask=mask,
                    features_1=mae_features,
                    features_2=simclr_features,
                )
            else:
                loss, loss_states, _ = self.loss(imgs=imgs, pred=pred, mask=mask)
            loss.backward()

            if self.step_count % self.steps_per_update == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

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

    def val(self, epoch, data_loader):
        """visualize the first 16 predicted images on val dataset"""
        self.model.eval()
        with torch.no_grad():
            val_img = next(iter(data_loader))["out_1"].to("cuda")
            predicted_val_img, mask, _ = self.model(val_img)
            predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
            img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
            img = rearrange(img, "(v h1 w1) c h w -> c (h1 h) (w1 v w)", w1=2, v=3)

        return {"imgs": (img + 1) / 2, "ACC@1": 0}
