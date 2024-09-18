import glob

import numpy as np

import torch
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
import carla
import cv2
import os



class CarlaDataset(Dataset):
    def __init__(self, data_dir):
        self.img_list=[]
        self.data_list=[]

        self.data_dir="./data/"
        self.img_list =  self.img_list+glob.glob(self.data_dir+'/*.jpg')
        self.data_list = self.data_list+glob.glob(self.data_dir+'/*.npy')
        self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    ])
        self.transform2= transforms.Resize((96,96),antialias=True)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # print(len(self.img_list))


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Load the RGB image and corresponding action. C = number of classes
        idx:      int, index of the data

        return    (image, action), both in torch.Tensor format
        """
        data =  np.load(self.data_list[idx], allow_pickle=True)
        img = read_image(self.img_list[idx])
        resized_image = img[:,:,80:560]
        image= self.transform2(resized_image)#resize the image
        normalized_image = self.normalize(image.float() / 255.0)
        actions = torch.Tensor(data[:2])#convert actions to tensor
        locations = torch.Tensor(data[3:])
        location_final=torch.Tensor(np.array((locations[0]-locations[2],locations[1]-locations[3])))
        return (normalized_image, actions,location_final)


def get_dataloader(data_dir, batch_size, num_workers=4):
    return torch.utils.data.DataLoader(
                CarlaDataset(data_dir),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True
            )