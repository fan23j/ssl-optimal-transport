import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Inference_Multilabel_Loss(nn.Module):
    def __init__(self, cfg):
        super(Inference_Multilabel_Loss, self).__init__()
        self.temperature = cfg.LOSS.TEMPERATURE
        self.gamma = cfg.DATASET.RATIO_GAMMA
        self.alpha = cfg.LOSS.SINKHORN_OT_PAIRWISE_ALPHA
        
#     def iterate_P(self, sim_matrix, b, num_iterations=15):
#         P = torch.exp(sim_matrix) # [samples, num_classs, 2]
       
#         for _ in range(num_iterations):
#             if torch.isnan(P).any():
#                 import pudb; pudb.set_trace()
#             sum_in = P.sum(dim=2) #[samples, num_class]
#             P = torch.div(P, sum_in.unsqueeze(2)) #[samples, num_class, 2] / [samples, num_class, 1]

#             sum_down = P.sum(dim=0)
#             P = torch.div(P, sum_down/b)
#         return P
    def calculate_M(self, Q_pos, S, alpha, k=0.01):
        M = torch.zeros_like(Q_pos).to(Q_pos.device)
        indices = torch.tensor(list(S.values()), dtype=torch.long)[:, :2].to(Q_pos.device)
        i1, i2 = indices[:, 0], indices[:, 1]

        for i in range(Q_pos.shape[0]):
            mask = Q_pos[i, i1] + Q_pos[i, i2] < alpha

            M[i, i1[mask]] += k
            M[i, i2[mask]] += k
    
        return M
                    
    def iterate_Q(self, sim_matrix, S, num_iterations=15):
        Q = torch.exp(sim_matrix) # [samples, num_classs, 2]
        print(len(S))
        for _ in range(num_iterations):
            sum_in = Q.sum(dim=2) #[samples, num_class]
            Q = torch.div(Q, sum_in.unsqueeze(2)) #[samples, num_class, 2] / [samples, num_class, 1]
            
            Q[:,:,0] = Q[:,:,0] * torch.exp(self.calculate_M(Q[:,:,0], S, self.alpha))
        return Q


    def forward(self, features, text_features, targets, dataset, sim_matrix_whole=None, total_num=0,**kwargs):
        sim = torch.matmul(features, text_features.t()) / self.temperature
        bs, num_class = sim.shape
        sim_matrix = torch.zeros((bs, num_class, 2))
        sim_matrix[:,:,0] = sim / 2
        sim_matrix[:,:,1] = -sim / 2

        loss = torch.tensor(0)
        if sim_matrix_whole is not None:
            #bs = total_num
            #b = torch.tensor(dataset.ratios).to("cuda")
            #b = torch.cat([b.unsqueeze(1), 1-b.unsqueeze(1)], dim=1)
            
            M = self.iterate_Q(sim_matrix_whole.to("cuda"), dataset.pairwise_ratios, num_iterations=8)

            return {"classify_anything_loss": loss}, M

        return {"classify_anything_loss": loss}, sim_matrix

    
