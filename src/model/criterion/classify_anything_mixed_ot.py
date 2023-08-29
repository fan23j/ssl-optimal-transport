import torch
import torch.nn.functional as F
import torch.nn as nn
from .asymmetric import AsymmetricLossOptimized


class Classify_Anything_Mixed_OT_Loss(nn.Module):
    def __init__(
        self,
        cfg,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=False,
    ):
        super(Classify_Anything_Mixed_OT_Loss, self).__init__()
        self.temperature = cfg.LOSS.TEMPERATURE
        self.asym_loss = AsymmetricLossOptimized(
            gamma_neg, gamma_pos, clip, eps, disable_torch_grad_focal_loss
        )

    def forward(self, features, text_features, targets, dataset_indices, **kwargs):
        """
        features: [B, 512]
        text_features: [num_class, 512]
        targets: [B, num_class]
        dataset_indices: [B]
        """
        sim_matrix = torch.matmul(features, text_features.t()) / self.temperature

        # init to numerator of softmax
        P0 = torch.exp(sim_matrix)

        # splitting data based on dataset_indices
        cifar_indices = dataset_indices == 1
        coco_indices = dataset_indices == 0

        coco_sim_matrix = sim_matrix[coco_indices.nonzero().squeeze()]
        cifar_sim_matrix = sim_matrix[cifar_indices.nonzero().squeeze()]

        # (number of multiclass images) * 1 + (number of multilabel images) * 0.1
        m = (cifar_indices).sum().item() + (coco_indices).sum().item() * 0.1

        P = iterate_P(P0, sim_matrix, m, 5)

        # Compute loss
        multiclass_loss = F.cross_entropy(P, targets.to("cuda"))
        multilabel_loss = self.asym_loss(sim_matrix, targets.to("cuda"))
        total_loss = multiclass_loss + multilabel_loss

        return {
            "classify_anything_loss": total_loss,
            "multiclass_loss": multiclass_loss,
            "multilabel_loss": multilabel_loss,
        }, [
            coco_sim_matrix,
            cifar_sim_matrix,
        ]


def iterate_P(P, sim_matrix, m, num_iterations=5):
    for _ in range(num_iterations):
        # C1
        column_sums_P = P.sum(dim=0)  # [num_class]
        scaling_factors = torch.min(
            sim_matrix / column_sums_P, torch.tensor(1.0).to(sim_matrix.device)
        )  # [batch_size, num_class]
        D = torch.diag(scaling_factors).unsqueeze(1)  # [batch_size, 1]
        P = P * D

        # C2
        total_sum = P.sum()
        scaling_factor = m / total_sum
        P = P * scaling_factor
    return P
