import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Inference_Sinkhorn_Loss(nn.Module):
    def __init__(self, cfg):
        super(Inference_Sinkhorn_Loss, self).__init__()
        self.temperature = cfg.LOSS.TEMPERATURE
        
    def sinkhorn_knopp(self, a, b, M, reg=0.5, numItermax=1000, stopThr=1e-9, verbose=False, log=False, warn=True, warmstart=None, **kwargs):
        """
        a: 1_bz
        b: ratios [r_1,r_2,...]
        M: Sim/Cost
        """

        # init data
        dim_a = a.shape[0]
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
    
    def iterate_P(self, sim_matrix, b, num_iterations=5):
        P = torch.exp(sim_matrix)

        for _ in range(num_iterations):
            sum_r = P.sum(dim=1)
            P = torch.div(P, sum_r.unsqueeze(1))
            
            sum_c = P.sum(dim=0)
            P = torch.div(P, sum_c.t()/b)
        return P


    def forward(self, features, text_features, targets, dataset, sim_matrix_whole=None, total_num=0,**kwargs):
        sim_matrix = torch.matmul(features, text_features.t()) / self.temperature
        loss = torch.tensor(0)
        if sim_matrix_whole is not None:
            bs = total_num
            b = torch.tensor(dataset.ratios).to("cuda") * bs
            M = self.iterate_P(sim_matrix_whole, b, num_iterations=10)
            return {"classify_anything_loss": loss}, M
        


        return {"classify_anything_loss": loss}, sim_matrix

    
