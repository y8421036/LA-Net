import os
import numpy as np
import copy

import torch



def check_dir_exist(dir):
    if os.path.exists(dir):
        return
    else:
        names = os.path.split(dir)
        dir = ''
        for name in names:
            dir = os.path.join(dir,name)
            if not os.path.exists(dir):
                try:
                    os.mkdir(dir)
                except:
                    pass
        print('dir','\''+dir+'\'','is created.')

def cal_Dice(img1,img2):
    shape = img1.shape
    I = 0
    U = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img1[i,j] >= 1 and img2[i,j] >= 1:
                I += 1
            if img1[i,j] >= 1 or img2[i,j] >= 1:
                U += 1
    return 2*I/(I+U+1e-5)

def cal_acc(img1,img2):
    shape = img1.shape
    acc = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img1[i,j] == img2[i,j]:
                acc += 1
    return acc/(shape[0]*shape[1])

def dice_loss(y_true, y_pred):
    smooth = 1
    intersection = torch.sum((y_true * y_pred)[:,1:,...])
    coeff = (2. *  intersection + smooth) / (torch.sum(y_true[:,1:,...]) + torch.sum(y_pred[:,1:,...]) + smooth)
    return (1. - coeff)

def cal_PRE_REC_JAC_BACC(pred, gt):
    eps = 1e-10  
    FP = np.float(np.sum((pred == 1) & (gt == 0)))
    FN = np.float(np.sum((pred == 0) & (gt == 1)))
    TP = np.float(np.sum((pred == 1) & (gt == 1)))
    TN = np.float(np.sum((pred == 0) & (gt == 0)))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    jaccard = TP / (TP + FN + FP) 

    TPR = recall
    TNR = TN/(TN+FP)
    balance_accuracy = (TPR+TNR)/2

    return precision, recall, jaccard, balance_accuracy