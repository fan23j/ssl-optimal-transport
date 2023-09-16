import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Inference_Sinkhorn_Loss(nn.Module):
    def __init__(self, cfg):
        super(Inference_Sinkhorn_Loss, self).__init__()
        self.temperature = cfg.LOSS.TEMPERATURE
        
    def sinkhorn_knopp(a, b, M, reg, numItermax=1000, stopThr=1e-9, verbose=False, log=False, warn=True, warmstart=None, **kwargs):
        """
        a: 1_bz
        b: ratios [r_1,r_2,...]
        M: Sim/Cost
        """

        # init data
        dim_a = len(a)
        dim_b = b.shape[0]

        u = torch.ones(dim_a, dtype=M.dtype, device=M.device) / dim_a
        v = torch.ones(dim_b, dtype=M.dtype, device=M.device) / dim_b

        K = torch.exp(M / (-reg))

        Kp = (1 / a).reshape(-1, 1) * K

        err = 1
        for ii in range(numItermax):
            uprev = u
            vprev = v
            KtransposeU = K.t() @ u
            v = b / KtransposeU
            u = 1. / (Kp @ v)

            if (torch.any(KtransposeU == 0)
                    or torch.any(torch.isnan(u)) or torch.any(torch.isnan(v))
                    or torch.any(torch.isinf(u)) or torch.any(torch.isinf(v))):
                u = uprev
                v = vprev
                break
            if ii % 10 == 0:
                tmp2 = torch.einsum('i,ij,j->j', u, K, v)
                err = torch.norm(tmp2 - b)  # violation of marginal

                if err < stopThr:
                    break

        return u.reshape((-1, 1)) * K * v.reshape((1, -1))

    def forward(self, features, text_features, targets, dataset, **kwargs):
        sim_matrix = torch.matmul(features, text_features.t()) / self.temperature
        cost_matrix = 1.0 - sim_matrix
        
        bs = features.shape[0]
        a = dataset.ratios * bs
        b = torch.ones(bs)
        
        M = self.sinkhorn_knopp(a, b, sim_matrix / cost_matrix, numItermax=100)

        return {"classify_anything_loss": loss}, M

    
