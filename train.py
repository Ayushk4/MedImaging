import torch 
import numpy as np
import torchvision
from torch import nn
from tqdm import tqdm
import loader
from datetime import datetime
import timm
import os
from utils import accuracy

MODEL_NAME = 'tf_efficientnet_b4_ns'
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
    prev_state_dict = torch.load('pretrained_models/' + MODEL_NAME + '.pt', map_location=torch.device('cpu'))

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
        raise 
    model.load_state_dict(prev_state_dict)
    model.classifier = nn.Sequential(
                nn.Linear(in_features=1792, out_features=625),
                nn.ReLU(), nn.Dropout(p=0.3),
                nn.Linear(in_features=625, out_features=256),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=2), 
            )
    return model.to(DEVICE)

def train_batch_loop(model, trainloader, optimizer, criterion):
    train_loss = 0.0
    train_acc = 0.0
    for images,labels,_ in tqdm(trainloader): 
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        logits = model(images)
        loss = criterion(logits,labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_acc += accuracy(logits,labels)
        
    return train_loss / len(trainloader), train_acc / len(trainloader) 

def valid_batch_loop(model, validloader, optimizer, criterion):
    valid_loss = 0.0
    valid_acc = 0.0
    
    for images,labels,_ in tqdm(validloader):
        
        # move the data to CPU
        images = images.to(DEVICE) 
        labels = labels.to(DEVICE)
        
        logits = model(images)
        loss = criterion(logits,labels)
        
        valid_loss += loss.item()
        valid_acc += accuracy(logits,labels)
        
    return valid_loss / len(validloader), valid_acc / len(validloader)

def fit(model, trainloader, validloader, optimizer, criterion, savefolder_):
    
    valid_min_loss = np.Inf 
    
    for i in range(EPOCHS):
        
        model.train()
        avg_train_loss, avg_train_acc = train_batch_loop(model, trainloader, optimizer, criterion)
        
        model.eval()
        avg_valid_loss, avg_valid_acc = valid_batch_loop(model, validloader, optimizer, criterion)
        
        if avg_valid_loss <= valid_min_loss :
            print("Valid_loss decreased {} --> {}".format(valid_min_loss,avg_valid_loss))
            torch.save(model.state_dict(), savefolder_ + '/saved.pt')
            valid_min_loss = avg_valid_loss

            
        print("Epoch : {} Train Loss : {:.6f} Train Acc : {:.6f}".format(i+1, avg_train_loss, avg_train_acc))
        print("Epoch : {} Valid Loss : {:.6f} Valid Acc : {:.6f}".format(i+1, avg_valid_loss, avg_valid_acc))


if __name__ == "__main__":
    trainloader, testloader, valloader = create_dataloaders()
    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    try:
        os.mkdir("savefolder")
    except:
        pass
    savefolder = "savefolder/" + str(datetime.now()).replace(" ", "_").replace(":", "_").split('.')[0]
    os.mkdir(savefolder)

    fit(model, trainloader, valloader, optimizer, criterion, savefolder)

    model.load_state_dict(torch.load(savefolder+'/saved.pt'))
    model.eval()
    mapper = {1: 'Cardiomegaly', 0: 'No Finding'}
    strs = ["imageID,disease"]
    preds = []
    for images, _, paths in tqdm(testloader): 
        images = images.to(DEVICE)
        
        logits = model(images)
        for p, l in zip(paths, logits.argmax(1).cpu().tolist()):
            strs.append(p.split('/')[-1] + "," + mapper[l])
    

    open(savefolder + "/sample_submission.csv", 'w+').write("\n".join(strs).strip())