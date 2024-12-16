import cv2
import numpy as np

import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import v2

from torch.utils.data import Dataset, DataLoader

sobel_y_1 = np.array([
  [-1, -1, -1],
  [0, 0, 0],
  [1, 1, 1],
])
sobel_y_2 = sobel_y_1 * -1

def getAugmentationEdges(image):

  grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  # Apply two filters, upper and lower for different bones
  edges_1 = cv2.filter2D(grayscaled, -1, sobel_y_1)
  edges_2 = cv2.filter2D(grayscaled, -1, sobel_y_2)

  # Cutout noisy background 
  _, edges_1 = cv2.threshold(edges_1, 10, 255, cv2.THRESH_TOZERO)
  _, edges_2 = cv2.threshold(edges_2, 10, 255, cv2.THRESH_TOZERO)

  edges = edges_1 + edges_2

  max_brightness = edges.max()

  # Normalize color
  edges = cv2.convertScaleAbs(edges, alpha = 255/max_brightness, beta = 0)

  return edges
  
transform_baseImage = v2.Compose([
  v2.Resize(256),
  v2.CenterCrop(256),
  v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )
])

transform_edgesImage = v2.Compose([
    v2.ToDtype(torch.float),
    v2.Normalize(mean=[0.449], std=[0.226])
])


class KneeOsteoarthritis_Edges(Dataset):
    def __init__(self, dataset):
        self.images = []
        self.edges_images = []
        self.labels = []
        
        for data in dataset:
            image = data[0]
            image_agmentation = image.numpy()*255
            image_agmentation = np.moveaxis(image_agmentation, 0, -1)
            edges_image = getAugmentationEdges(image_agmentation)
            # print(image.shape, image_agmentation.shape, edges_image.shape)
            edges_image = torch.tensor(edges_image)
            label = data[1]
            
            image = transform_baseImage(image)
            # print(image.shape, edges_image.shape)
            edges_image = transform_edgesImage(edges_image.unsqueeze(0))
            
            self.images.append(image)
            self.edges_images.append(edges_image)            
            self.labels.append(label)
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        edges_image = self.edges_images[idx]
        label = self.labels[idx]
        
        return image, edges_image, label

def get_KneeOsteoarthritis_Edges(path) -> KneeOsteoarthritis_Edges:
  
  transform_toTensor = transforms.Compose([transforms.ToTensor()])
  
  # Downloading raw dataset
  dataset = torchvision.datasets.ImageFolder(path, transform_toTensor)
  
  # Building Augmented Dataset
  return KneeOsteoarthritis_Edges(dataset)
