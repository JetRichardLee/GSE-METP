# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 17:09:42 2022

@author: 31271
"""

import random
import torch
import numpy as np
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F
from graph_env import GraphEnv
from dygcn import DyGCN #,model_train,model_predict
from ASTGCN_r import make_model
import pandas as pd
from utilss import scaled_Laplacian, cheb_polynomial
from utils import get_auc,get_f1
GCN_INPUT_SIZE = 274
GCN_WEIGHT_SIZE = 256
LSTM_HIDDEN_SIZE = 256
HISTORICAL_LEN = 10

dygcn = DyGCN(GCN_INPUT_SIZE, GCN_WEIGHT_SIZE, LSTM_HIDDEN_SIZE, HISTORICAL_LEN)

sequence = torch.from_numpy(np.load('contact_x.npy'))
target = torch.from_numpy(np.load('contact_y.npy'))

allone = torch.from_numpy(np.ones((1,274,274),dtype=np.float32))

new_sequence  = torch.from_numpy(np.zeros((5,10,274,274),dtype=np.float32))
new_target  = torch.from_numpy(np.zeros((5,274,274),dtype=np.float32))

for i in range(5):
	new_sequence[i]=sequence
	new_target[i]=target
	for times in range(10):
		a = random.randint(0,273)
		b = random.randint(0,273)
		if a==b:
			continue
		c = random.randint(0, 1)
		for l in range(10):
			new_sequence[i,l,a,b]=c
			new_sequence[i,l,b,a]=c
		new_target[i,a,b]=c
		new_target[i,b,a]=c
		#df=np.delete(df,a,0)


np.save(file="haggle-s.npy", arr=new_sequence)
np.save(file="haggle-t.npy", arr=new_target)


class DYGCNLoss(nn.Module):
	def __init__(self, BETA=1.5):
		super(DYGCNLoss, self).__init__()
		self.BETA = BETA

	def forward(self, x, y):
		if self.BETA <= 1.:
			self.BETA = 1.5

		y = y.reshape(-1)
		x = x.reshape(-1)

		p = y * (self.BETA - 1) + 1
		loss = torch.sum(torch.mul(p, (y - x).pow(2)), dim=-1)
		return loss
	
def model_train(model,sequence,target,allone,epoch=20,lr=1e-3,beta = 1.2):

	#sequence = sequence.permute(1,2,0).unsqueeze(0)
	print(sequence.shape)
	loss_function = DYGCNLoss(beta)

	# loss_function = nn.BCELoss()

	optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=0.0005)

	loss_list = []
	auc_list = []

	fpr_list = []
	tpr_list = []

	device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
	for e in range(epoch):

		x = model(sequence.to(device))
		diag = torch.triu(x) - torch.triu(x, 1)
		x = x - diag
		print(x.sum())
		optimizer.zero_grad()
		loss = loss_function(x,target.to(device))
		loss.backward()
		optimizer.step()

		#auc,f,t = get_auc(x,target)
		auc,f1 = get_f1(x,target.to(device),allone.to(device),0.5)

		loss_list.append(loss.item())
		auc_list.append(auc)

		print('epoch = ',e+1,'    ','f1 = :', f1, '   ', 'loss = :', loss.item())

	torch.save(model.state_dict(), 'haggle-s.pkl')

def model_predict(model,sequence,target,allone,threshold=0.5):

	#sequence = sequence.permute(1,2,0).unsqueeze(0)
	x = model(sequence)
	model.eval()

	diag = torch.triu(x) - torch.triu(x, 1)
	x = x - diag

	#auc,_,_ = get_auc(x,target,threshold)
	auc, f1 = get_f1(x,target,allone,threshold)

	return auc,f1
def train_model(epochs,lr,allone):
	for eposide in range(epochs):
		model_train(dygcn,new_sequence[0],new_target[0],allone,1,lr)
		model_train(dygcn,new_sequence[1],new_target[1],allone,1,lr)
		model_train(dygcn,new_sequence[2],new_target[2],allone,1,lr)
		model_train(dygcn,new_sequence[3],new_target[3],allone,1,lr)
		model_train(dygcn,new_sequence[4],new_target[4],allone,1,lr)
        
train_model(10,1e-4,allone)
for i in range(30):
	print(i)
	dygcn.load_state_dict(torch.load('haggle-s.pkl'))
	train_model(5,1e-5,allone)
	dygcn.load_state_dict(torch.load('haggle-s.pkl'))
	train_model(5,1e-4,allone)
	dygcn.load_state_dict(torch.load('haggle-s.pkl'))
	train_model(5,1e-3,allone)
	dygcn.load_state_dict(torch.load('haggle-s.pkl'))
	train_model(5,1e-4,allone)
	dygcn.load_state_dict(torch.load('haggle-s.pkl'))
	train_model(5,1e-5,allone)