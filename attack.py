#!/usr/bin/env python  
# encoding: utf-8  

import random
import torch
import numpy as np
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from graph_env import GraphEnv
from setup import dygcn,new_sequence,new_target,ATTACK_PARAM,HISTORICAL_LEN#,alls

import warnings

device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

class EmbeddingBuffer:
	def __init__(self,capacity):
		self.capacity = capacity
		self.buffer = []
		self.position = 0

	def push(self,hidden,action,reward,done):
		if len(self.buffer) < self.capacity:
			self.buffer.append(None)
		self.buffer[self.position] = (hidden,action,reward,done)
		self.position = int((self.position + 1) % self.capacity) # ring buffer

	def sample(self,batch_size):
		batch = random.sample(self.buffer,batch_size)   # adjust?
		hidden,action,reward,done = map(np.stack,zip(*batch))
		return hidden,action,reward,done

	def __len__(self):
		return len(self.buffer)
	
	
	
class features(nn.Module):
	def __init__(self,num_inputs,out_features,weight=None):
        
		super(features,self).__init__()
		self.out_features = out_features
		self.num_inputs = num_inputs
		self.weight = Parameter(torch.FloatTensor(num_inputs, out_features))
		if weight!=None:
			self.weight.data = weight
		else:
			stdv = 1. / math.sqrt(self.weight.size(1))
			self.weight.data.uniform_(-stdv, stdv)
		self.sig= nn.Sigmoid()
		self.feature_size = num_inputs*out_features
		
	def forward(self, seq):
		
		length = seq.shape[1]
		seq = seq.reshape(-1,self.num_inputs)
		fvector = torch.matmul(seq, (self.weight)).reshape(-1,length,self.feature_size)
		return fvector
		
class EmbeddingNetwork(nn.Module):
	def __init__(self,feature_size,embedding_size):

		super(EmbeddingNetwork,self).__init__()
		self.feature_size = feature_size
		self.relu=nn.ReLU()
		
		self.lstm1 = nn.LSTM(self.feature_size,embedding_size,1,batch_first = True)
		self.lstm2 = nn.LSTM(self.feature_size,embedding_size,1,batch_first = True)
		for name,para in self.lstm1.named_parameters():
			nn.init.normal_(para, 0.0001, std=0.001)
		for name,para in self.lstm2.named_parameters():
			nn.init.normal_(para, 0.0001, std=0.001)
		self.bn1 = nn.BatchNorm1d(1)
		self.bn2 = nn.BatchNorm1d(1)
		
	def forward(self, fvector):

		_, (embedding1, _) = self.lstm1(fvector)
		embedding1 = self.bn1(embedding1.permute(1,0,2))

		_, (embedding2, _) =self.lstm2(fvector)
		embedding2 = self.bn2(embedding2.permute(1,0,2))
		state = torch.cat([embedding1,embedding2],dim = 1)
		#state = torch.squeeze(state)
		return state
	
class PolicyNetwork(nn.Module):
	def __init__(self, num_actions,feature_size,hidden_size,action_range,init_w=3e-2,log_std_min=-20,log_std_max=2):
		super(PolicyNetwork,self).__init__()
		self.log_std_min = log_std_min
		self.log_std_max = log_std_max
		self.hidden = hidden_size
		self.feature = feature_size
		self.embedding = EmbeddingNetwork(feature_size,hidden_size)
		
		self.bn1 = nn.BatchNorm1d(1)
		self.bn2 = nn.BatchNorm1d(1)
		self.sig=nn.Sigmoid()
		self.linear1 = nn.Linear(hidden_size, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.linear3 = nn.Linear(hidden_size, hidden_size)
		self.linear4 = nn.Linear(hidden_size, hidden_size)
		self.linear5 = nn.Linear(hidden_size, hidden_size)
		self.linear6 = nn.Linear(hidden_size, hidden_size)
		self.linear7 = nn.Linear(hidden_size, hidden_size)
		self.linear8 = nn.Linear(hidden_size, hidden_size)
		for name,para in self.linear1.named_parameters():
			nn.init.normal_(para, 0.5, std=0.1)
		for name,para in self.linear5.named_parameters():
			nn.init.normal_(para, 0.5, std=0.1)

		self.mean_linear = nn.Linear(hidden_size, num_actions)
		self.mean_linear.weight.data.uniform_(-init_w, init_w)
		self.mean_linear.bias.data.uniform_(-init_w, init_w)

		self.log_std_linear = nn.Linear(hidden_size, num_actions)
		self.log_std_linear.weight.data.uniform_(-init_w, init_w)
		self.log_std_linear.bias.data.uniform_(-init_w, init_w)

		self.action_range = action_range
		self.num_actions = num_actions

        
        
	def forward(self,hidden):
		
		state = self.embedding(hidden)
		x1 = F.relu(self.linear1(state[:,0,:]))
		x1 = F.relu(self.linear2(x1))
		x1 = F.relu(self.linear3(x1))
		x2 = F.relu(self.linear5(state[:,1,:]))
		x2 = F.relu(self.linear6(x2))
		x2 = F.relu(self.linear7(x2))
		x1 = torch.stack([x1],dim=1)
		x2 = torch.stack([x2],dim=1)
		x1 = self.bn1(x1)
		x2 = self.bn2(x2)
		x = torch.cat([x1,x2],dim = 1)


		mean = (self.mean_linear(x))

		log_std = self.log_std_linear(x)
		log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

		return mean, log_std
	
	def getAction(self,hidden):

		mean,log_std = self.forward(hidden)

		std = log_std.exp()

		normal = Normal(0,1)

		z = normal.sample()
		action = self.action_range*torch.sigmoid(mean)#+std*z)  #
		action = action.cpu().detach().numpy()
		return action


	def evaluate(self,hidden,epsilon=1e-6):
		mean, log_std = self.forward(hidden)
		std = log_std.exp()  # no clip in evaluation, clip affects gradients flow

		normal = Normal(0, 1)
		z = normal.sample()
		action_0 = torch.sigmoid(mean)# + std * z)  # TanhNormal distribution as actions; reparameterization trick
		action = self.action_range * action_0

		return action, z, mean, log_std


class Queue:
	def __init__(self,capacity):
		self.capacity = capacity
		self.buffer = []
		self.position = 0

	def push(self,hidden,action,reward,next_state,done):
		if len(self.buffer) < self.capacity:
			self.buffer.append(None)
		
		self.buffer[self.position] = (hidden,action,reward,next_state,done)
		self.position = int((self.position + 1) % self.capacity) # ring buffer

	def get(self):
		(hidden,action,reward,next_state,done) = self.buffer[self.position] 
		return hidden,action,reward,next_state,done
	def getrewards(self):
		rewards=0
		t=1
		for i in range(5):
			(hidden,action,reward,next_state,done) = self.buffer[(self.position + i) % self.capacity] 
			rewards+=reward*t
			t*=0.7
		return rewards
	def qsize(self):
		return len(self.buffer)
	
class SoftQNetwork(nn.Module):
	def __init__(self,num_inputs,num_actions,feature_size, hidden_size,init_w=3e-3):
		super(SoftQNetwork,self).__init__()
		self.hidden = hidden_size
		self.feature = feature_size
		self.embedding = EmbeddingNetwork(feature_size,hidden_size)
		self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.linear3 = nn.Linear(hidden_size, hidden_size)
		self.linear4 = nn.Linear(hidden_size, 1)
		self.init_w = init_w
		self.linear4.weight.data.uniform_(-init_w, init_w)
		self.linear4.bias.data.uniform_(-init_w, init_w)
		self.sig=nn.Sigmoid()

	def forward(self, hidden, action):
		
		state = self.embedding(hidden)
		action=action.reshape(1500,2,-1)
		x = torch.cat([state, action], 2)
		x = F.relu(self.linear1(x))
		x = F.relu(self.linear2(x))
		x = F.relu(self.linear3(x))
		x = self.linear4(x)

		x = torch.sum(x,dim=1)

		return x
	


class Agent:
	def __init__(self,embedding_buffer,hidden_dim,action_range):
		self.embedding_buffer = embedding_buffer

		self.featurelize=features(adj_dim,fea_dim)
		self.soft_q_net1 = SoftQNetwork(128, action_dim, self.featurelize.feature_size, hidden_dim)
		self.soft_q_net2 = SoftQNetwork(128, action_dim, self.featurelize.feature_size, hidden_dim)

		self.policy_net = PolicyNetwork(action_dim, self.featurelize.feature_size, hidden_dim, action_range,
		                                log_std_min=ATTACK_PARAM['log_std_min'],
		                                log_std_max=ATTACK_PARAM['log_std_max'])

		self.soft_q_criterion1 = nn.MSELoss()
		self.soft_q_criterion2 = nn.MSELoss()
		self.policy_criterion = nn.MSELoss()


		soft_q_lr = ATTACK_PARAM['soft_q_lr']
		policy_lr = ATTACK_PARAM['policy_lr']

		self.soft_q_optimizer1 = torch.optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
		self.soft_q_optimizer2 = torch.optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
		self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=policy_lr)
	def getHidden(self,seq):
		hidden = self.featurelize(seq)
		return hidden
	def getAction(self, hidden):
		action = self.policy_net.getAction(hidden)
		action = action[0]
		return action

	def embedding(self,batch_size,tag,reward_scale=10,auto_entropy=True,target_entropy=-2,gamma=0.95,soft_tau=1e-2):
		hidden, action, reward, done = self.embedding_buffer.sample(batch_size)

		hidden = torch.squeeze(torch.FloatTensor(hidden)).to(device)
		#print(state.size())
		#print(next_state.size())
		#reward = torch.FloatTensor(reward).to(device)
		action = torch.FloatTensor(action).to(device)
		reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        
		predicted_q_value1 = self.soft_q_net1(hidden, action)
		predicted_q_value2 = self.soft_q_net2(hidden, action)
        
		if tag == True:        
			new_action, z, mean, log_std = self.policy_net.evaluate(hidden.detach())
		else:        
			new_action, z, mean, log_std = self.policy_net.evaluate(hidden)

		reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6)


		# ========== Training Q Function
		
		target_q_value = reward
		q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
		q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

		self.soft_q_optimizer1.zero_grad()
		q_value_loss1.backward()
		self.soft_q_optimizer1.step()
		
		self.soft_q_optimizer2.zero_grad()
		q_value_loss2.backward()
		self.soft_q_optimizer2.step()

		# ========== Training Policy Function
		predicted_new_q_value = torch.min(self.soft_q_net1(hidden.detach(), new_action.detach()), self.soft_q_net2(hidden.detach(), new_action.detach()))
		predicted_q_value = torch.min(self.soft_q_net1(hidden.detach(), action), self.soft_q_net2(hidden.detach(), action.detach()))
		decrease = predicted_new_q_value-predicted_q_value
		decrease[decrease>0]=0     
		policy_loss = (0 - self.policy_criterion(new_action,action)*decrease).mean()
		self.policy_optimizer.zero_grad()
		policy_loss.backward()
		self.policy_optimizer.step()

		return predicted_new_q_value.mean()

if __name__ == '__main__':
	
	device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
	for testtimes in range(10):
		f=open('log_times{}.txt'.format(testtimes),'w')

		warnings.filterwarnings("ignore")
		metr = ATTACK_PARAM['metric']
	
		# initial the dynamic graph Env
		adj_dim = 274
		fea_dim = 4
		
			
		env = GraphEnv(dygcn, new_sequence[0], new_target[0],ATTACK_PARAM['num_U'],ATTACK_PARAM['attack_graph_max'],ATTACK_PARAM['attack_r'],metric=metr)
		
		state_dim = env.state_num
		action_dim = env.action_num
		
		# initial the attack Agent
		action_range = state_dim - 1 # the range of the index of node
		embedding_buffer = EmbeddingBuffer(ATTACK_PARAM['embedding_buffer_size'])
		CrAtt = Agent(embedding_buffer,int(ATTACK_PARAM['hidden_dim']),action_range)
		CrAtt.featurelize.to(device)
		CrAtt.policy_net.to(device)
		CrAtt.soft_q_net1.to(device)
		CrAtt.soft_q_net2.to(device)
		#CrAtt.policy_net=torch.load("C:/Users/31271/Desktop/pro/code/policy")
	
		metric_list = []
		episode_rewards = []
		tag=True
		attack_method = 'Random'
		random_episode = ATTACK_PARAM['random_episode']
		best = 100
		rec=np.zeros(200,np.float32)
		training_index = [0,1,2,3,4]
		for training in range(1):
			for i_episode in range(50):

				sequence = new_sequence[training_index[(i_episode%5)]]
				target = new_target[training_index[(i_episode%5)]]
				env = GraphEnv(dygcn, sequence, target,ATTACK_PARAM['num_U'],ATTACK_PARAM['attack_graph_max'],ATTACK_PARAM['attack_r'],metric=metr)
		
				nowbest=100
				f.write('Episode{}'.format(i_episode).center(100,'=')+'\n')
				seq,state,metric= env.reset()
                    
				hidden = CrAtt.getHidden(seq.to(device))
			
				f.write(" --- Fraction of attack nodes : {}".format(ATTACK_PARAM['attack_r']) + "\n" +
			      " --- Number of attack graphs  : {} / {}".format(ATTACK_PARAM['attack_graph_max'],HISTORICAL_LEN) + "\n" +
			      " --- {} on clean data : {} ".format(metr,metric) + "\n")
				attack_metric = 0
				episode_reward = 0
				cnt = 0
				#			params = list(CrAtt.policy_net.lstm1.named_parameters())
				print("Now is episode{}: best is {}".format(i_episode, best))
				q=Queue(5)
				torch.cuda.empty_cache()
				torch.cuda.empty_cache()
				torch.cuda.empty_cache()
				torch.cuda.empty_cache()
				torch.cuda.empty_cache()
        
				for step in range(ATTACK_PARAM['max_steps']):
					#if i_episode>4:
					torch.cuda.empty_cache()
					torch.cuda.empty_cache()
					torch.cuda.empty_cache()
					torch.cuda.empty_cache()
					torch.cuda.empty_cache()
					if attack_method == 'Random' and len(embedding_buffer) > ATTACK_PARAM['embedding_size']:
						attack_method = 'Embedding'
					if i_episode==0:
						tag = False
                    
					# ===== sample an action and execute it =====

					#print("getac")
					action = CrAtt.getAction(hidden.to(device))

					f.write( "[ori ac : {}]   \n".format(action))
					if attack_method == 'Random' or step%8 == 0 or cnt>10 or i_episode <5:
						action[0,0]= random.randint(0,273)
						action[0,1]= random.randint(0,273)
						action[1,0]= random.randint(0,273)
						action[1,1]= random.randint(0,273)
				#print("getstep")
					torch.cuda.empty_cache()
					torch.cuda.empty_cache()
					torch.cuda.empty_cache()
					torch.cuda.empty_cache()
					torch.cuda.empty_cache()
					next_seq,next_state,reward,done,attack_metric,progress = env.step(action)
				#print("gethidden")
	
					next_hidden = CrAtt.getHidden(next_seq)
			
				# === long time no reward , stop
					if reward <= 0:
						cnt += 1
					else:
						cnt = 0


					if cnt >= ATTACK_PARAM['batch_size']/2:
						print("\n"+'--- Warning : reward is too sparse'+"\n")
						break
	
				# === print info
					f.write(
						     "[Attack method : {}]   \n".format(attack_method) +
						      "[Attack step : {}]   \n".format(step) +
						       "[{} : {}]   \n".format(metr,attack_metric) +
						        "[reward : {}]  \n".format(reward) +
						     "[Attack edges : {}]    \n".format(progress))
	
				# ===== update buffer,state,and episode reward =====
					f.write(str(i_episode))
					f.write(":"+'\n')
				#print(str(i_episode))
				#print(":"+'\n')
				#print(state)
				             
					q.push(hidden.cpu().detach(),action,reward,next_hidden.cpu().detach(),done)
					nowbest = min(nowbest, attack_metric)
					hidden = next_hidden
					seq = next_seq
					f.write(str(action))
					torch.cuda.empty_cache()
					torch.cuda.empty_cache()
					torch.cuda.empty_cache()
					torch.cuda.empty_cache()
					torch.cuda.empty_cache()
					if q.qsize()==5:
						hi,ac,re,ne,do = q.get()
						rewards = q.getrewards()
						if (re-0>1e-2):
							embedding_buffer.push(hi,ac,re,do)
					#elif (rewards+re-0<-1e-2):
					#	replay_buffer.push(se,ac,rewards+re,ne,do)
						elif (re-0<-1e-2):
							embedding_buffer.push(hi,ac,re,do)    
						f.write("put rewards:"+'\n')
						f.write(str(rewards)+'\n')
						rewards-=re/15
				
				# ===== update parameters of CrAtt =====
					if attack_method == 'Embedding'  and training<=3:
						for i in range(ATTACK_PARAM['update_itr']):
							CrAtt.embedding(batch_size=ATTACK_PARAM['embedding_size'],
						                 tag=tag,
						                 reward_scale=ATTACK_PARAM['reward_scale'],
						                 auto_entropy=True,
						                 target_entropy=-1.*action_dim*2)
	
				# ===== stop attack =====
					if done == 1:
						f.write("\n" + " --- Warning : The attack edges has reached its limit---"+'\n')
						f.write("best is {}".format(best))
						f.write("record is {}".format(rec))
						break
				rec[i_episode]=nowbest
				best=min(best,nowbest)    
				f.write("best is {}".format(best))
				print(best)
                
		f.close()
