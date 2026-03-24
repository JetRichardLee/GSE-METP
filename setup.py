#!/usr/bin/env python  
# encoding: utf-8  

from utils import *
from dygcn import *
from ASTGCN_r import *
import numpy as np
import pandas as pd
# ================= initial the DyGCN ==================
GCN_INPUT_SIZE = 274 
GCN_WEIGHT_SIZE = 256
LSTM_HIDDEN_SIZE = 256
HISTORICAL_LEN = 10

dygcn = DyGCN(GCN_INPUT_SIZE, GCN_WEIGHT_SIZE, LSTM_HIDDEN_SIZE, HISTORICAL_LEN)
dygcn.load_state_dict(torch.load('haggle-s.pkl'))

# ================= initial a sample =============
#alls= np.load(file="as.npy")
#alls = torch.from_numpy(alls)
#print(alls.shape)
new_sequence = np.load(file="haggle-s.npy")
new_target = np.load(file="haggle-t.npy")
new_sequence = torch.from_numpy(new_sequence)
new_target = torch.from_numpy(new_target)
device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

# ========== set the parameters of attack agent ==========
ATTACK_PARAM = {
	'replay_buffer_size' : 10000,  # adjust : decrease?        (default : 1e6)
	'embedding_buffer_size':10000,
	'hidden_dim' : 128,          # adjust?  less is stable?  (default: 128)
	'update_itr' : 2,            #                           (default : 1)
	'random_episode' : 10,        # random attack episode     (default : 2)
	'max_episodes' : 60,
	'max_steps' : 20000,

	'batch_size' : 1500,         # adjust :
	'embedding_size' : 1500,         # adjust :

	'reward_scale' : 200.,        # adjust : increase?  (default : 20)

	'soft_q_lr' : 1e-3,         # adjust?
	'policy_lr' : 1e-3,
	'alpha_lr'  : 3e-4,

	'log_std_min' : -20,        # adjust?
	'log_std_max' : 2,          # adjust?

	'num_U' : 274,               # U adjust?

	'attack_graph_max' : [6,8,10][2],            # m (max m = HISTORICAL_LEN)
	'attack_r' : [0.01,0.02,0.03][1],            # r (fraction of attack nodes)

	'metric' : ['f1_score'][0]
}
