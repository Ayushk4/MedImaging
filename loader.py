import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import folder
import numpy as np
import os
from tqdm.notebook import tqdm
import csv
import random
import numpy as np
SPLIT_RATIO = (6,7)
IMG_SIZE    = 224
np.random.seed(49)
random.seed(49)
torch.manual_seed(49)

label_to_idx = {'cardiomegaly': 1}

def get_label(path):
    data = {}
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                pass
                line_count += 1
            else:
                data[row[0]] = "".join(row[1].split()).lower()
                line_count += 1
        assert line_count == len(data) + 1
    return data

class ImageDataset(data.Dataset):
    def __init__(self, data, isTrain):
        self.data = data
        self.isTrain = isTrain
        self.loader = folder.default_loader

        if self.isTrain:
            self.transform = T.Compose([
                                T.Resize(size=(IMG_SIZE,IMG_SIZE)), # Resizing the image to be 224 by 224
                                T.RandomRotation(degrees=(-20,+20)), #Randomly Rotate Images by +/- 20 degrees, Image argumentation for each epoch
                                T.ToTensor(), #converting the dimension from (height,weight,channel) to (channel,height,weight) convention of PyTorch
                                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) # Normalize by 3 means 3 StD's of the image net, 3 channels
                            ])
        else:
            self.transform = T.Compose([
                                T.Resize(size=(IMG_SIZE,IMG_SIZE)), # Resizing the image to be 224 by 224
                                T.ToTensor(), #converting the dimension from (height,weight,channel) to (channel,height,weight) convention of PyTorch
                                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) # Normalize by 3 means 3 StD's of the image net, 3 channels
                            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        sample = self.loader(path)
        sample = self.transform(sample)

        return sample, label_to_idx.get(label, 0), path
    
def pad_fn(batch):
    images = torch.tensor([batch[i][0] for i in range(len(batch))])
    labels = torch.LongTensor([batch[i][1] for i in range(len(batch))])
    paths = [batch[i][2] for i in range(len(batch))]
    return images, labels, paths

train_and_val_data = [('dataset_cardiomegaly/train/'+k, v) for k,v in get_label('dataset_cardiomegaly/train.csv').items()]
random.shuffle(train_and_val_data)
split_idx  = (len(train_and_val_data) * SPLIT_RATIO[0])//SPLIT_RATIO[1]

train_dataset = ImageDataset(train_and_val_data[:split_idx], True)
val_dataset   = ImageDataset(train_and_val_data[split_idx:], False)

test_dataset  = ImageDataset([('dataset_cardiomegaly/test/'+k, v) for k,v in get_label('dataset_cardiomegaly/test.csv').items()], False)

