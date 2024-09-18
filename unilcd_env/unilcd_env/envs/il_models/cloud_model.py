import torch
import torch.nn as nn

# # Create a new model without these layers
class CloudModel(nn.Module):
    def __init__(self):
        super(CloudModel, self).__init__()
        # Add your model class for use
        pass
    
    def forward(self, x,locations):
        # Add the forward pass for your model below
        pass


# # Example of cloud model
# import torch
# import torch.nn as nn
# import timm


# # # Create a new model without these layers
# class CloudModel(nn.Module):
#     def __init__(self):
#         super(CloudModel, self).__init__()
#         self.model = timm.create_model('regnety_002', pretrained=True)
#         self.features=nn.Sequential(*list(self.model.children())[:-1])
#         self.goal=nn.Sequential(
#             nn.Linear(2, int(self.model.head.fc.in_features/2)),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Linear(int(self.model.head.fc.in_features/2),int(self.model.head.fc.in_features))
#         )
#         self.lin =nn.Sequential(
#             nn.Linear(self.model.head.fc.in_features*2, 512),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Linear(256, 2)
#         )
#     def forward(self, x,locations):
#         x=self.model.stem(x)
#         x=self.model.s1(x)
#         x=self.model.s2(x)
#         x=self.model.s3(x)
#         x=self.model.s4(x)
#         x=self.model.final_conv(x)
#         x=self.model.head.global_pool(x)
#         y=self.goal(locations)
#         sf=torch.cat((x, y), dim=1)
#         sf = self.lin(sf)
#         return sf
