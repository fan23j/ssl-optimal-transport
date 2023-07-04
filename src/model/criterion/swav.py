import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist


class SwAV_Loss(nn.Module):
    def __init__(self, cfg):
        super(SwAV_Loss, self).__init__()
        self.epsilon = cfg.LOSS.SWAV_EPSILON
        self.temperature = cfg.LOSS.TEMPERATURE
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.nmb_crops = cfg.DATASET.MULTICROP_NMB_CROPS
        self.rank = cfg.TRAIN.LOCAL_RANK if cfg.TRAIN.DISTRIBUTE else 1
        self.sinkhorn_iterations = cfg.LOSS.SINKHORN_MAX_ITER
        self.crops_for_assign = cfg.TRAIN.CROPS_FOR_ASSIGN
        self.distributed = cfg.TRAIN.DISTRIBUTE

    @torch.no_grad()
    def distributed_sinkhorn(self, out):
        Q = torch.exp(
            out / self.epsilon
        ).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * self.rank  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if self.distributed:
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(self.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if self.distributed:
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(self, output, queue, model, embedding):
        loss = 0

        for i, crop_id in enumerate(self.crops_for_assign):
            with torch.no_grad():
                out = output[
                    self.batch_size * crop_id : self.batch_size * (crop_id + 1)
                ].detach()

                # time to use the queue
                if queue is not None:
                    if not torch.all(queue[i, -1, :] == 0):
                        out = torch.cat(
                            (
                                torch.mm(
                                    queue[i],
                                    model.module.prototypes.weight.t(),
                                ),
                                out,
                            )
                        )
                    # fill the queue
                    queue[i, self.batch_size :] = queue[i, : -self.batch_size].clone()
                    queue[i, : self.batch_size] = embedding[
                        crop_id * self.batch_size : (crop_id + 1) * self.batch_size
                    ]

                # get assignments
                q = self.distributed_sinkhorn(out)[-self.batch_size :]
                if torch.isnan(q).any():
                    import pudb

                    pudb.set_trace()

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
                x = (
                    output[self.batch_size * v : self.batch_size * (v + 1)]
                    / self.temperature
                )
                subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            loss += subloss / (np.sum(self.nmb_crops) - 1)
        loss /= len(self.crops_for_assign)
        return {"swav_loss": loss}
