import torch
import torch.nn.functional as F
import torch.nn as nn
from .asymmetric import AsymmetricLossOptimized
from .utils import convert_targets


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

    def forward(
        self,
        features,
        multilabel_text_features,
        multiclass_text_features,
        targets,
        dataset_indices,
        dataset,
        **kwargs
    ):
        """
        features: [B, 512]
        text_features: [num_class, 512]
        targets: [B, num_class]
        dataset_indices: [B]
        """
        multilabel_sim_matrix = (
            torch.matmul(features, multilabel_text_features.t()) / self.temperature
        )
        multiclass_sim_matrix = (
            torch.matmul(features, multiclass_text_features.t()) / self.temperature
        )

        # splitting data based on dataset_indices
        multiclass_indices = dataset_indices == 1
        multilabel_indices = dataset_indices == 0

        _multilabel_sim_matrix = multilabel_sim_matrix[
            multilabel_indices.nonzero().squeeze()
        ]
        _multiclass_sim_matrix = multiclass_sim_matrix[
            multiclass_indices.nonzero().squeeze()
        ]

        # (number of multiclass images) * 1 + (number of multilabel images) * 0.1
        m = (multiclass_indices).sum().item() + (multilabel_indices).sum().item() * 0.1

        P = iterate_P(multiclass_sim_matrix, m, 5)

        multiclass_targets, multilabel_targets = convert_targets(
            targets,
            dataset.mixed_indices,
            dataset.multilabel_labels,
            dataset.multiclass_labels,
            dataset_indices,
        )
        
        # Compute loss
        multilabel_loss = self.asym_loss(
            multilabel_sim_matrix, multilabel_targets.to("cuda")
        )

        multiclass_loss = -torch.sum(multiclass_targets.to("cuda") * torch.log(P))

        total_loss = multiclass_loss + multilabel_loss

        return {
            "loss": total_loss,
            "multiclass_loss": multiclass_loss,
            "multilabel_loss": multilabel_loss,
        }, [
            _multilabel_sim_matrix,
            _multiclass_sim_matrix,
            multilabel_targets,
            multiclass_targets,
        ]


# def iterate_P(sim_matrix, m, num_iterations=5):
#     P = torch.exp(sim_matrix)

#     for _ in range(num_iterations):
#         # C1
#         row_sums_P = P.sum(dim=1)
#         scaling_factors = torch.min(
#             torch.ones_like(row_sums_P) / row_sums_P,
#             torch.tensor(1.0).to(P.device),
#         )
#         D = torch.diag(scaling_factors)
#         P = torch.matmul(D, P)

#         # C2
#         total_sum = P.sum()
#         scaling_factor = m / total_sum
#         P = P * scaling_factor
#     return P

def iterate_M(sim_matrix, b, num_iterations=5):
        P = torch.exp(sim_matrix)

        for _ in range(num_iterations):
            sum_in = P.sum(dim=2)
            P = torch.div(P, sum_in.unsqueeze(2))

            sum_down = P.sum(dim=0)
            P = torch.div(P, sum_down/b)
        return P

def iterate_P(sim_matrix, m, num_iterations=5):
    P = torch.exp(sim_matrix)

    for _ in range(num_iterations):
        # C1
        row_sums_P = P.sum(dim=1)
        scaling_factors = torch.max(
            row_sums_P,
            torch.tensor(1.0).to(P.device),
        )
        P = torch.div(P, scaling_factors.unsqueeze(1))

        P = P * m / P.sum()
    return P
import torch
import torch.nn.functional as F
import torch.nn as nn
from .asymmetric import AsymmetricLossOptimized
from .utils import convert_targets


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

    def forward(
        self,
        features,
        multilabel_text_features,
        multiclass_text_features,
        targets,
        dataset_indices,
        dataset,
        **kwargs
    ):
        """
        features: [B, 512]
        text_features: [num_class, 512]
        targets: [B, num_class]
        dataset_indices: [B]
        """
        multilabel_sim_matrix = (
            torch.matmul(features, multilabel_text_features.t()) / self.temperature
        )
        multiclass_sim_matrix = (
            torch.matmul(features, multiclass_text_features.t()) / self.temperature
        )

        # splitting data based on dataset_indices
        multiclass_indices = dataset_indices == 1
        multilabel_indices = dataset_indices == 0

        _multilabel_sim_matrix = multilabel_sim_matrix[
            multilabel_indices.nonzero().squeeze()
        ]
        _multiclass_sim_matrix = multiclass_sim_matrix[
            multiclass_indices.nonzero().squeeze()
        ]

        # (number of multiclass images) * 1 + (number of multilabel images) * 0.1
        m = (multiclass_indices).sum().item() + (multilabel_indices).sum().item() * 0.1

        P = iterate_P(multiclass_sim_matrix, m, 5)

        multiclass_targets, multilabel_targets = convert_targets(
            targets,
            dataset.mixed_indices,
            dataset.multilabel_labels,
            dataset.multiclass_labels,
            dataset_indices,
        )
        
        # Compute loss
        multilabel_loss = self.asym_loss(
            multilabel_sim_matrix, multilabel_targets.to("cuda")
        )

        multiclass_loss = -torch.sum(multiclass_targets.to("cuda") * torch.log(P))

        total_loss = multiclass_loss + multilabel_loss

        return {
            "loss": total_loss,
            "multiclass_loss": multiclass_loss,
            "multilabel_loss": multilabel_loss,
        }, [
            _multilabel_sim_matrix,
            _multiclass_sim_matrix,
            multilabel_targets,
            multiclass_targets,
        ]


# def iterate_P(sim_matrix, m, num_iterations=5):
#     P = torch.exp(sim_matrix)

#     for _ in range(num_iterations):
#         # C1
#         row_sums_P = P.sum(dim=1)
#         scaling_factors = torch.min(
#             torch.ones_like(row_sums_P) / row_sums_P,
#             torch.tensor(1.0).to(P.device),
#         )
#         D = torch.diag(scaling_factors)
#         P = torch.matmul(D, P)

#         # C2
#         total_sum = P.sum()
#         scaling_factor = m / total_sum
#         P = P * scaling_factor
#     return P

def iterate_M(sim_matrix, b, num_iterations=5):
        P = torch.exp(sim_matrix)

        for _ in range(num_iterations):
            sum_in = P.sum(dim=2)
            P = torch.div(P, sum_in.unsqueeze(2))

            sum_down = P.sum(dim=0)
            P = torch.div(P, sum_down/b)
        return P

def iterate_P(sim_matrix, m, num_iterations=5):
    P = torch.exp(sim_matrix)

    for _ in range(num_iterations):
        # C1
        row_sums_P = P.sum(dim=1)
        scaling_factors = torch.max(
            row_sums_P,
            torch.tensor(1.0).to(P.device),
        )
        P = torch.div(P, scaling_factors.unsqueeze(1))

        P = P * m / P.sum()
    return P
