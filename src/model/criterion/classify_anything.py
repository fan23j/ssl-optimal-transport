import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class Classify_Anything_Loss(nn.Module):
    def __init__(self, cfg):
        super(Classify_Anything_Loss, self).__init__()
        self.temperature = cfg.LOSS.TEMPERATURE

    def forward(self, features, text_features, targets, **kwargs):
        """
        features: [B, 300]
        text_features: [num_class, 512]
        targets: [B]
        """
        # Normalize features and labels_vector along the feature dimension
        # features_norm = F.normalize(features, dim=1)
        # labels_vector_norm = F.normalize(labels_vector, dim=1)
        import pudb; pudb.set_trace()
        cosim_matrix = torch.matmul(features.t(), text_features[targets.item()].unsqueeze(0)) / self.temperature
        correlations_2d = TSNE(n_components=2).fit_transform(cosim_matrix.cpu().numpy())

        # Plotting
        plt.scatter(correlations_2d[:, 0], correlations_2d[:, 1], c='b', alpha=0.5)

        plt.title('Visualization of Local Class Embedding and Dense Image Feature')
        plt.savefig('/home/ubuntu/ssl-optimal-transport/runs/test/images/fig.png')
        import pudb; pudb.set_trace()
        
        cosim_softmax = F.softmax(cosim_matrix, dim=1)

        loss = F.cross_entropy(cosim_matrix, targets)

        return {"classify_anything_loss": loss}, cosim_softmax
