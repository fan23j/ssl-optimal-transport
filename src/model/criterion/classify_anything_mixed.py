import torch
import torch.nn.functional as F
import torch.nn as nn
from .asymmetric import AsymmetricLossOptimized


class Classify_Anything_Mixed_Loss(nn.Module):
    def __init__(
        self,
        cfg,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=False,
    ):
        super(Classify_Anything_Mixed_Loss, self).__init__()
        self.temperature = cfg.LOSS.TEMPERATURE
        self.weight_decay = cfg.TRAIN.WD
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

        cosim_matrix = torch.matmul(features, text_features.t()) / self.temperature

        # Splitting data based on dataset_indices
        multiclass_indices = dataset_indices == 1
        multilabel_indices = dataset_indices == 0

        multilabel_targets = targets[multilabel_indices.nonzero().squeeze().to("cpu")]
        multiclass_targets = targets[multiclass_indices.nonzero().squeeze().to("cpu")]

        multilabel_cosim_matrix = cosim_matrix[multilabel_indices.nonzero().squeeze()]
        multiclass_cosim_matrix = cosim_matrix[multiclass_indices.nonzero().squeeze()]

        # Compute the multilabel loss using asym_loss
        multilabel_loss = self.asym_loss(multilabel_cosim_matrix, multilabel_targets.to("cuda"))

        # Convert multiclass one-hot targets to class labels for cross_entropy
        multiclass_labels = torch.argmax(multiclass_targets, dim=1)
        multiclass_loss = F.cross_entropy(multiclass_cosim_matrix, multiclass_labels.to("cuda"))

        total_loss = multilabel_loss + multiclass_loss

        return {
            "classify_anything_loss": total_loss,
            "multilabel_loss": multilabel_loss,
            "multiclass_loss": multiclass_loss,
        }, [
            multilabel_cosim_matrix,
            multiclass_cosim_matrix,
        ]
