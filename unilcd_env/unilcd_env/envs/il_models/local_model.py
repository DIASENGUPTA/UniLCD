import torch
import torch.nn as nn


class LocalModel(nn.Module):
    def __init__(self):
        super(LocalModel, self).__init__()
        # Add your model below.

    def forward(self, observation, locations, return_features=False):
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

# #Example of local model
# import torch
# import torch.nn as nn
# import timm


# class LocalModel(torch.nn.Module):
#     def __init__(self):
#         """
#         Implementation of the network layers. The image size of the input
#         observations is 96x96 pixels.
#         """
#         super(LocalModel, self).__init__()
#         self.model = timm.create_model('regnety_002', pretrained=True)
#         self.goal = nn.Sequential(
#             nn.Linear(2, 12),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Linear(12,24)
#         )

#         self.lin = nn.Sequential(
#             nn.Linear(48, 512),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Linear(256, 2)
#         )

#     def forward(self, observation, locations, return_features=False):
#         """
#         The forward pass of the network. Returns the prediction for the given
#         input observation.
#         observation:   torch.Tensor of size (batch_size, height, width, channel)
#         return         torch.Tensor of size (batch_size, C), features
#         """
#         x = self.model.stem(observation)
#         x = self.model.s1(x)
#         x = self.model.final_conv(x)
#         x = self.model.head.global_pool(x)
#         y = self.goal(locations)
#         sf = torch.cat((x, y), dim=1)
#         x = self.lin(sf)
#         if return_features:
#             return x, sf
#         else:
#             return x, None