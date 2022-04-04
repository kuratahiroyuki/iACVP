
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import torch.nn as nn
from metrics import cofusion_matrix
import math
import numpy as np    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def CBLoss(labels, logits, beta, no_of_classes):
    
    labels_cls = labels.clone().detach().cpu().numpy()
    labels_cls = labels_cls.reshape(labels_cls.shape[0],).astype(np.int32)

    samples_per_cls = [len(labels_cls)-sum(labels_cls), sum(labels_cls)]
    
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    
    #print('effective_num', np.array(effective_num))
    
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes
        
    labels_one_hot = F.one_hot(labels.squeeze().long().to(device),num_classes=no_of_classes).float()
        
    weights = torch.tensor(weights).float().cuda()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.size()[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)

    criterion = nn.BCELoss(weight = weights.cuda())
    cb_loss = criterion(input = logits, target = labels.float())
     
    return cb_loss


