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

image_id_2_location = {}
for im in os.listdir('dataset_cardiomegaly/train'):
    image_id_2_location[im] = 'dataset_cardiomegaly/train/'+im
for im in os.listdir('dataset_cardiomegaly/test'):
    image_id_2_location[im] = 'dataset_cardiomegaly/test/'+im
for im in os.listdir('dataset_full/sample/images'):
    image_id_2_location[im] = 'dataset_full/sample/images/'+im
for im in os.listdir('dataset_full/sample/sample/images'):
    image_id_2_location[im] = 'dataset_full/sample/sample/images/'+im

diseases = ['Effusion', 'Nodule', 'Fibrosis', 'Cardiomegaly', 'Mass', 'Pneumothorax', 'Edema', 'Consolidation', 'Pneumonia', 'Hernia', 'Infiltration', 'Pleural_Thickening', 'Atelectasis', 'Emphysema']
label_to_idx = {x.lower():i for i, x in enumerate(diseases)}
idx_to_label = {i:x.lower() for i, x in enumerate(diseases)}
idx_to_label_uppercased = {i:x for i, x in enumerate(diseases)}

def get_label(path, old_dataset_file=False):
    data = {}
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                pass
                line_count += 1
            else:
                data[row[0]] = ("".join(row[1].split()).lower().split('|'), old_dataset_file)
                line_count += 1
        assert line_count == len(data) + 1
    return data

class ImageDataset(data.Dataset):
    def __init__(self, data, isTrain, isTest=False):
        self.data = data
        self.isTrain = isTrain
        self.isTest = isTest
        self.loader = folder.default_loader

        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        import timm

        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        config = resolve_data_config({}, model=model)
        self.transform = create_transform(**config)
        if False:# self.isTrain:
            self.transform = T.Compose([
                                T.Resize(size=(IMG_SIZE,IMG_SIZE)), # Resizing the image to be 224 by 224
                                T.RandomRotation(degrees=(-20,+20)), #Randomly Rotate Images by +/- 20 degrees, Image argumentation for each epoch
                                T.ToTensor(), #converting the dimension from (height,weight,channel) to (channel,height,weight) convention of PyTorch
                                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) # Normalize by 3 means 3 StD's of the image net, 3 channels
                            ])
        else:
            pass#self.transform = T.Compose([
            #                    T.Resize(size=(IMG_SIZE,IMG_SIZE)), # Resizing the image to be 224 by 224
            #                    T.ToTensor(), #converting the dimension from (height,weight,channel) to (channel,height,weight) convention of PyTorch
            #                    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) # Normalize by 3 means 3 StD's of the image net, 3 channels
            #                ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, (label, from_old_dataset) = self.data[idx]
        indexed_label = torch.zeros(14)
        if not self.isTest:
            if 'nofinding' in label:
                assert label == ["nofinding"], label
            else:
                for l in label:
                    indexed_label[label_to_idx[l]] = 1

        if from_old_dataset:
            task_padding = torch.zeros(14)
            task_padding[label_to_idx['cardiomegaly']] = 1
        else:
            task_padding = torch.ones(14)

        sample = self.loader(path)
        sample = self.transform(sample)

        return sample, indexed_label, path, task_padding
    
def pad_fn(batch):
    images = torch.tensor([batch[i][0].tolist() for i in range(len(batch))])
    labels = torch.LongTensor([batch[i][1].tolist() for i in range(len(batch))])
    paths  = [batch[i][2] for i in range(len(batch))]
    task_pads = torch.LongTensor([batch[i][3].tolist() for i in range(len(batch))])
    return images, labels, paths, task_pads

train_and_val_data = get_label('dataset_cardiomegaly/train.csv', True)

for k, v in get_label('dataset_full/14_train.csv').items():
    train_and_val_data[k] = v
train_and_val_data = [(image_id_2_location[k], (v1,v2)) for k,(v1,v2) in train_and_val_data.items()]
random.shuffle(train_and_val_data)
split_idx  = (len(train_and_val_data) * SPLIT_RATIO[0])//SPLIT_RATIO[1]

train_dataset = ImageDataset(train_and_val_data[:split_idx], True)
val_dataset   = ImageDataset(train_and_val_data[split_idx:], False)

test_dataset  = ImageDataset([(image_id_2_location[k], v) for k,v in get_label('dataset_cardiomegaly/test.csv').items()], False, True)

