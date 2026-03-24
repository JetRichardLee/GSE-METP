#!/usr/bin/env python  
# encoding: utf-8  

import numpy as np
import os
import torch
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from matplotlib.pyplot import MultipleLocator

def normalize_adj(adj):

    input_size = adj.shape[0]
    # normalize
    # adj = adj + torch.eye(self.in_features)
    degree_diag = torch.diag(adj.sum(0))
    diag_ = torch.pow(degree_diag, -1).flatten()
    diag_[torch.isinf(diag_)] = 0.
    adj = torch.mm(diag_.reshape(input_size,input_size), adj)

    return adj


def get_auc(x,y,allone,thresh=0.5):

    """disabled"""
    acc = 0
    return acc

def get_f1(x,y,allone,thresh=0.5):

    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    x = torch.where(x>thresh,1,0)
    TP = torch.sum(x*y).cpu()
    TN = torch.sum((allone-x)*(allone-y)).cpu()
    FN = torch.sum((allone-x)*y).cpu()
    FP = torch.sum(x*(allone-y)).cpu()
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    f1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)
    acc = acc.numpy()
    f1=f1.numpy()


    return acc,f1

def model_predict(model,sequence,target,allone,threshold=0.5):
	#print("forward")
	device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
	sequence=sequence.to(device)
	print(sequence.shape)   
	sequence = sequence.reshape(sequence.shape[0] * sequence.shape[1], sequence.shape[1])
	x = model(sequence)
	model.eval()
	#print("endforward")
	diag = torch.triu(x) - torch.triu(x, 1)
	x = x - diag
	target = target.to(device)
    

	#auc = get_auc(x,target,allone,threshold)
	#auc = 0
	auc,f1 = get_f1(x,target,allone,threshold)
	print(auc,f1)

	return auc,f1