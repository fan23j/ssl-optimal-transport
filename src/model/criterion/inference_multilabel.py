import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Inference_Multilabel_Loss(nn.Module):
    def __init__(self, cfg):
        super(Inference_Multilabel_Loss, self).__init__()
        self.temperature = cfg.LOSS.TEMPERATURE
        
    def iterate_P(self, sim_matrix, b, num_iterations=5):
        P = torch.exp(sim_matrix)

        for _ in range(num_iterations):
            sum_in = P.sum(dim=2)
            P = torch.div(P, sum_in.unsqueeze(2))

            sum_down = P.sum(dim=0)
            P = torch.div(P, sum_down/b)
        return P


    def forward(self, features, text_features, targets, dataset, sim_matrix_whole=None, total_num=0,**kwargs):
        sim = torch.matmul(features, text_features.t()) / self.temperature
        bs, num_class = sim.shape
        sim_matrix = torch.zeros((bs, num_class, 2))
        sim_matrix[:,:,0] = sim / 2
        sim_matrix[:,:,1] = -sim / 2
        
        loss = torch.tensor(0)
        if sim_matrix_whole is not None:
            bs = total_num
            b = torch.tensor(dataset.ratios).to("cuda") * bs
            b = torch.cat([b.unsqueeze(1), b.unsqueeze(1)], dim=1)
            M = self.iterate_P(sim_matrix_whole.to("cuda"), b, num_iterations=10)
            return {"classify_anything_loss": loss}, M

        return {"classify_anything_loss": loss}, sim_matrix

    
