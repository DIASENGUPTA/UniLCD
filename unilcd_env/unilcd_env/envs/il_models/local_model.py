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

#Example of local model
# import torch
# import torch.nn as nn
# import numpy as np
# import timm


# class LocalModel(torch.nn.Module):
#     def __init__(self):
#         """
#         Implementation of the network layers. The image size of the input
#         observations is 96x96 pixels.
#         """
#         super(LocalModel, self).__init__()
#         self.mobile = timm.create_model('mobilenetv2_100',pretrained=True)
#         self.feature_extractor=nn.Sequential(*list(self.mobile.children())[:-1])
#         self.mobile.classifier =nn.Sequential(
#             nn.Linear(self.mobile.classifier.in_features+2, 2))

#     def forward(self, observation, locations):
#         """
#         The forward pass of the network. Returns the prediction for the given
#         input observation.
#         observation:   torch.Tensor of size (batch_size, height, width, channel)
#         return         torch.Tensor of size (batch_size, C)
#         """
#         x=self.feature_extractor(observation)
#         x=torch.cat((x, locations), dim=1)
#         x = self.mobile.classifier(x)
#         return x