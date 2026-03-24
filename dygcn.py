#!/usr/bin/env python  
# encoding: utf-8  

import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import get_auc,get_f1
import math
from torch.nn.parameter import Parameter

class DyGCN(nn.Module):
	def __init__(self, nfeat,ngcn_out,nhid,seq_len = 10,n_layer=2,dropout=0.5):
		super(DyGCN, self).__init__()

		self.dropout = dropout
		self.nfeat = nfeat
		self.ngcn_out = ngcn_out
		self.seq_len = seq_len

		self.gc1 = GcnUnit(nfeat,ngcn_out)
		# self.gc2 = GcnUnit(nhid1, ngcn_out)
		self.lstm = LstmUnit(ngcn_out,nhid,n_layer,nfeat)

	def forward(self,adj):
		# adj : adjacency matrix
		#print(adj)
		x = F.relu(self.gc1(adj))
		x = F.dropout(x, self.dropout, training=self.training)

		# (10,274,256)
		x = x.reshape(self.seq_len,self.nfeat,self.ngcn_out)

		x = self.lstm(x)

		# return F.log_softmax(x, dim=1)
		return x

class GcnUnit(nn.Module):
	"""
	Simple GCN layer
	"""
	def __init__(self, in_features, out_features, bias=True):
		super(GcnUnit, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.FloatTensor(in_features, out_features))
		if bias:
			self.bias = Parameter(torch.FloatTensor(out_features))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self,adj):
		#print(adj.size())
		#print(self.weight.size())
		output = torch.matmul(adj, (self.weight))

		if self.bias is not None:
			return output + self.bias
		else:
			return output

	def __repr__(self):
		return self.__class__.__name__ + ' (' \
			   + str(self.in_features) + ' -> ' \
			   + str(self.out_features) + ')'

class LstmUnit(nn.Module):
	def __init__(self, in_dim, hidden_dim,n_layer,n_classes):
		super(LstmUnit, self).__init__()
		self.n_layer = n_layer
		self.hidden_dim = hidden_dim
		self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer)
		self.classifier = nn.Linear(hidden_dim, n_classes)

	def forward(self, x):
		seq_len = x.shape[0]
		out, (h_n, c_n) = self.lstm(x)
		# h_n = torch.squeeze(h_n)[self.n_layer - 1]
		out = out[seq_len - 1]
		x = self.classifier(out)
		x = torch.relu(x)
		# sigmoid lead to the grad disappeared
		# x = torch.sigmoid(x)
		return x

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

def model_train(model,sequence,target,epoch=20,lr=1e-3,beta = 1.2):

	loss_function = DYGCNLoss(beta)

	# loss_function = nn.BCELoss()

	optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=0.0005)

	loss_list = []
	auc_list = []

	fpr_list = []
	tpr_list = []

	for e in range(epoch):

		x = model(sequence)
		diag = torch.triu(x) - torch.triu(x, 1)
		x = x - diag

		optimizer.zero_grad()
		loss = loss_function(x,target)
		loss.backward()
		optimizer.step()

		auc = get_auc(x,target)

		loss_list.append(loss.item())
		auc_list.append(auc)

		print('epoch = ',e+1,'    ','auc = :', auc, '   ', 'loss = :', loss.item())

	torch.save(model.state_dict(), 'as.pkl')

def model_predict(model,sequence,target,allone,threshold=0.5):
	#print("forward")
	device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
	sequence=sequence.to(device)
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

