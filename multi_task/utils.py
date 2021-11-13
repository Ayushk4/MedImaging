import torch
import torch.nn.functional as F 

def accuracy(y_pred, y_true, padding="NotGiven"):
    if padding != "NotGiven":
        return accuracy_multitask(y_pred, y_true, padding)
    y_pred = F.softmax(y_pred, dim=1)
    top_p,top_class = y_pred.topk(1, dim=1)
    equals = top_class == y_true.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))

def accuracy_multitask(y_pred, y_true, padding):
    total = [torch.sum(padding[:, disease_idx]).item() for disease_idx in range(14)]
    preds = [sum([(round(p) == round(t))*pad
                for p, t, pad in zip(y_pred[:, disease_idx].tolist(),
                                y_true[:, disease_idx].tolist(),
                                padding[:, disease_idx].tolist()
                                )])
                for disease_idx in range(14)
            ]
    return preds, total
