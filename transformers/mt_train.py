import random
import torch 
import numpy as np
import torchvision
from torch import nn
from tqdm import tqdm
import mt_loader as loader
from datetime import datetime
import timm
import os
from utils import accuracy

MODEL_NAME = 'vit_base_patch16_224'#'tf_efficientnet_b4_ns'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5
LEARNING_RATE = 0.0005
BATCH_SIZE = 16
PRETRAINED = True

def create_dataloaders():
    trainloader = torch.utils.data.DataLoader(loader.train_dataset,
                                        batch_size=BATCH_SIZE,
                                        num_workers=1,
                                        shuffle=True
                                    )
    print("No. of batches in trainloader:{}".format(len(trainloader)))
    print("No. of Total examples:{}".format(len(trainloader.dataset)))

    valloader = torch.utils.data.DataLoader(loader.val_dataset,
                                        batch_size=BATCH_SIZE,
                                        num_workers=1,
                                        shuffle=False
                                    )
    print("No. of batches in valloader:{}".format(len(valloader)))
    print("No. of Total examples:{}".format(len(valloader.dataset)))

    testloader = torch.utils.data.DataLoader(loader.test_dataset,
                                        batch_size=BATCH_SIZE,
                                        num_workers=1,
                                        shuffle=False
                                    )
    print("No. of batches in testloader:{}".format(len(testloader)))
    print("No. of Total examples:{}".format(len(testloader.dataset)))
    return trainloader, testloader, valloader

def create_model():
    model = timm.create_model(MODEL_NAME, pretrained=True)
    if PRETRAINED != True:
        return model.to(DEVICE)
    #prev_state_dict = torch.load('pretrained_models/' + MODEL_NAME + '.pt', map_location=torch.device('cpu'))

    if MODEL_NAME == 'tf_efficientnet_b4_ns':
        del prev_state_dict['classifier.0.weight']
        del prev_state_dict['classifier.0.bias']
        del prev_state_dict['classifier.3.weight']
        del prev_state_dict['classifier.3.bias']
        del prev_state_dict['classifier.5.weight']
        del prev_state_dict['classifier.5.bias']
        prev_state_dict['classifier.weight'] = torch.randn(model.classifier.weight.shape)
        prev_state_dict['classifier.bias'] = torch.randn(model.classifier.bias.shape)
    else:
        pass
    #model.load_state_dict(prev_state_dict)
    model.head = nn.Sequential(
                nn.Linear(in_features=768, out_features=625),
                nn.LeakyReLU(), nn.Dropout(p=0.3),
                nn.Linear(in_features=625, out_features=256),
                nn.LeakyReLU(),
                nn.Linear(in_features=256, out_features=14),
                nn.Sigmoid()
            )
    return model.to(DEVICE)

def train_batch_loop(model, trainloader, optimizer, criterion):
    train_loss = 0.0
    correct_preds = torch.LongTensor([0 for _ in range(14)])
    num_totals = torch.LongTensor([0 for _ in range(14)])
    for images,labels,_,task_pad in tqdm(trainloader): 
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        task_pad = task_pad.to(DEVICE)
        
        logits = model(images)
        loss = criterion(logits, labels, task_pad)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        xxx, yyy = accuracy(logits, labels, task_pad)
        correct_preds += torch.LongTensor(xxx)
        num_totals += torch.LongTensor(yyy)
        
    return train_loss / len(trainloader), [round(p/t,3) if t != 0 else 0 for p, t in zip(correct_preds.tolist(), num_totals.tolist())]

def valid_batch_loop(model, validloader, optimizer, criterion):
    valid_loss = 0.0
    correct_preds = torch.LongTensor([0 for _ in range(14)])
    num_totals = torch.LongTensor([0 for _ in range(14)])
    for images,labels,_,task_pad in tqdm(validloader):
        
        # move the data to CPU
        images = images.to(DEVICE) 
        labels = labels.to(DEVICE)
        task_pad = task_pad.to(DEVICE)
        
        logits = model(images)
        loss = criterion(logits,labels,task_pad)
        
        valid_loss += loss.item()
        xxx, yyy = accuracy(logits, labels, task_pad)
        correct_preds += torch.LongTensor(xxx)
        num_totals += torch.LongTensor(yyy)

    return valid_loss / len(validloader), [round(p/t,3) if t != 0 else 0.0 for p, t in zip(correct_preds.tolist(), num_totals.tolist())]

def fit(model, trainloader, validloader, optimizer, criterion, savefolder_):
    
    valid_max_acc = 0
    
    for i in range(EPOCHS):
        
        model.train()
        avg_train_loss, avg_train_acc = train_batch_loop(model, trainloader, optimizer, criterion)
        
        model.eval()
        avg_valid_loss, avg_valid_acc = valid_batch_loop(model, validloader, optimizer, criterion)
        
        if avg_valid_acc[3] >= valid_max_acc :
            print("Valid_acc increased {} --> {}".format(valid_max_acc, avg_valid_acc[3]))
            torch.save(model.state_dict(), savefolder_ + '/saved.pt')
            valid_max_acc = avg_valid_acc[3]

            
        print("Epoch : {} Train Loss : {:.6f} Train Acc : {}".format(i+1, avg_train_loss, avg_train_acc))
        print("Epoch : {} Valid Loss : {:.6f} Valid Acc : {}".format(i+1, avg_valid_loss, avg_valid_acc))


if __name__ == "__main__":
    trainloader, testloader, valloader = create_dataloaders()
    model = create_model()

    bce_loss_object = nn.BCELoss(reduction='none')
    criterion = lambda logits, labels, pad: (bce_loss_object(logits, labels) * pad).sum()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    try:
        os.mkdir("savefullfolder")
    except:
        pass
    savefolder = "savefullfolder/" + str(datetime.now()).replace(" ", "_").replace(":", "_").split('.')[0]
    os.mkdir(savefolder)

    fit(model, trainloader, valloader, optimizer, criterion, savefolder)

    model.load_state_dict(torch.load(savefolder+'/saved.pt'))
    model.eval()
    mapper = loader.idx_to_label_uppercased
    strs = ["imageID,disease"]
    preds = []
    for images, _, paths, _ in tqdm(testloader):
        images = images.to(DEVICE)

        logits = model(images)
        for p, l in zip(paths, logits.cpu().tolist()):
            found_diseases = [mapper[i] for i, x in enumerate(l) if round(x) == 1]
            if len(found_diseases) == 0:
                found_diseases = ["No Finding"]
            strs.append(p.split('/')[-1] + "," + '|'.join(found_diseases))

    open(savefolder + "/sample_submission.csv", 'w+').write("\n".join(strs).strip())


