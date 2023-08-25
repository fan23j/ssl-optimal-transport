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
        cifar_indices = dataset_indices == 1
        coco_indices = dataset_indices == 0

        coco_targets = targets[coco_indices.nonzero().squeeze().to("cpu")]
        cifar_targets = targets[cifar_indices.nonzero().squeeze().to("cpu")]

        coco_cosim_matrix = cosim_matrix[coco_indices.nonzero().squeeze()]
        cifar_cosim_matrix = cosim_matrix[cifar_indices.nonzero().squeeze()]

        # Compute the COCO loss using asym_loss
        coco_loss = self.asym_loss(coco_cosim_matrix, coco_targets.to("cuda"))

        # Convert CIFAR one-hot targets to class labels for cross_entropy
        cifar_labels = torch.argmax(cifar_targets, dim=1)
        cifar_loss = F.cross_entropy(cifar_cosim_matrix, cifar_labels.to("cuda"))

        total_loss = coco_loss + cifar_loss

        return {
            "classify_anything_loss": total_loss,
            "coco_loss": coco_loss,
            "cifar_loss": cifar_loss,
        }, [
            coco_cosim_matrix,
            cifar_cosim_matrix,
        ]
