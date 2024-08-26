import torch
import torch.nn as nn


class LocalModel(nn.Module):
    def __init__(self):
        super(LocalModel, self).__init__()
        # Add your model below.

    def forward(self, observation, locations):
        """
        Add the forward pass for your model below. 
        You should also return the embeddings from your feature extractor
        along with the final output. This is required for switching.
        
        Example:
        ft = self.feature_extractor(observation)
        x = torch.cat((ft, locations), dim=1)
        x = self.classifier(x)
        return x, ft
        """
        pass