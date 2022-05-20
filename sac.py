import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
import matplotlib.pyplot as plt
import random
import gc
import pandas as pd
class ReplayBuffer(object):
    def __init__(self, mem_size, input_shape, n_actions, mem_cntr=0):
        self.mem_size = mem_size
        self.n_actions = n_actions
        self.mem_cntr = mem_cntr
        #self.state_memory = np.zeros((self.mem_size,*input_shape))
       
        #self.new_state_memory = np.zeros((self.mem_size,*input_shape))
        #self.action_memory = np.zeros((self.mem_size,n_actions))
        #self.reward_memory = np.zeros(self.mem_size)
        #self.terminal_memory = np.zeros(self.mem_size,dtype=np.bool_)
        self.data_path = "D:\\opensimrlmemory\\memory.csv"
        column_names = ['state_memory','new_state_memory','action_memory','reward_memory','terminal_memory']
        self.file = pd.DataFrame(columns=column_names,dtype=object)
        #self.file.to_csv(self.data_path,mode='w',index=False)
        

    def store_transition(self, state, action, reward,new_state, done):
        index = self.mem_cntr % self.mem_size
        #print(index,state,len(self.state_memory))
        """ self.state_memory[index]= state
        self.new_state_memory[index]= new_state
        self.action_memory[index]= action #action is an array, so array of arrays
        self.reward_memory[index]= reward
        self.terminal_memory[index]= done """
        #print('memcntr',self.mem_cntr)

        if self.mem_cntr < self.mem_size:
            data_buff = {'state_memory':state,'new_state_memory':new_state,'action_memory':action.tolist(),'reward_memory':reward,'terminal_memory':done}
            self.file.append(data_buff,ignore_index=True).to_csv(self.data_path,mode='a',index=False,header=False)
        else:
            self.file.loc[index,'state_memory':'terminal_memory'] = state,new_state,action.tolist(),reward,done
        self.mem_cntr += 1
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr,self.mem_size) # cntr can be greater than size, but don't want to take mems that don't exist
        """ batch = np.random.choice(max_mem,batch_size) # from 0 to max_mem of size batch_size
        
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminals = self.terminal_memory[batch] """
        skip = sorted(random.sample(range(1,max_mem+1),max_mem-batch_size))
        memory = pd.read_csv(self.data_path,skiprows=skip)
        states = []
        new_states = []
        actions = []
        rewards = []
        terminals = []
        for i in range(batch_size-1):
            state_temp = memory['state_memory'].tolist()[i]
            states.append(eval(state_temp))
            new_states_temp = memory['new_state_memory'].tolist()[i]
            new_states.append(eval(new_states_temp))
            actions_temp = memory['action_memory'].tolist()[i]
            actions.append(eval(actions_temp))
            rewards.append(np.float(memory['reward_memory'].tolist()[i]))
            terminals.append(np.bool(memory['terminal_memory'].tolist()[i]))
        
        return np.array(states),np.array(actions),np.array(rewards),np.array(new_states),np.array(terminals)

            


        


        
        return states,actions,rewards,new_states,terminals

    def save_memory(self):
        
        """ np.save('D:\\opensimrlmemory\\state_memory.npy',self.state_memory)
        np.save('D:\\opensimrlmemory\\new_state_memory.npy',self.new_state_memory)
        np.save('D:\\opensimrlmemory\\action_memory.npy',self.action_memory)
        np.save('D:\\opensimrlmemory\\reward_memory.npy',self.reward_memory)
        np.save('D:\\opensimrlmemory\\terminal_memory.npy',self.terminal_memory) """
        #np.save('D:\\opensimrlmemory\\mem_cntr.npy',np.array([self.mem_cntr]))
        #print(self.mem_cntr,'save')

        data_buff = {'state_memory':self.state_memory,'new_state_memory':self.new_state_memory,'action_memory':self.action_memory,'reward_memory':self.reward_memory,'terminal_memory':self.terminal_memory}
        self.file.append(data_buff,ignore_index=True).to_csv(self.data_path,mode='a',index=False,header=False)
    def load_memory(self,done=False):
        """ if done:
            del self.state_memory
            gc.collect()
            del self.new_state_memory
            gc.collect()
            del self.action_memory
            gc.collect()
            del self.reward_memory
            gc.collect()
            del self.terminal_memory
            gc.collect()
            print('mem deleted') """

       
        """  self.state_memory=np.load('D:\\opensimrlmemory\\state_memory.npy')
        self.new_state_memory = np.load('D:\\opensimrlmemory\\new_state_memory.npy')
        self.action_memory = np.load('D:\\opensimrlmemory\\action_memory.npy')
        self.reward_memory = np.load('D:\\opensimrlmemory\\reward_memory.npy')
        self.terminal_memory = np.load('D:\\opensimrlmemory\\terminal_memory.npy') """
        #self.mem_cntr = np.load('D:\\opensimrlmemory\\mem_cntr.npy')[0]
        #print('here',self.mem_cntr)

        #print(len(self.state_memory))
        
        data = pd.read_csv('D:\\opensimrlmemory\\memory.csv')

        self.state_memory=np.array(data['state_memory'])
        self.new_state_memory = np.array(data['new_state_memory'])
        self.action_memory = np.array(data['action_memory'])
        self.reward_memory = np.array(data['reward_memory'])
        self.terminal_memory = np.array(data['terminal_memory'])
class CriticNetwork(nn.Module):
    '''this evaluates the value of a state,action pair'''
    def __init__(self,beta, input_dims, n_actions,fc1_dims=300, fc2_dims=300,
                name='Critic_Network',chkpt_dir='C:\\Users\\wilian\\Desktop\\Booker\\RL'):
        super(CriticNetwork,self).__init__()
        self.beta = beta #learning rate
        self.input_dims= input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, name +'_sac')
        print(f"critic: {self.chkpt_file}")
        self.fc1 = nn.Linear(self.input_dims[0]+n_actions,self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    def forward(self,state, action):
        action_value = self.fc1(T.cat([state,action],dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)
        q = self.q(action_value)
        return q
    def save_checkpoint(self):
        print('.....saving checkpoint.....')
        T.save(self.state_dict(), self.chkpt_file)
    def load_checkpoint(self):
        print('.......loading checkpoint........')
        self.load_state_dict(T.load(self.chkpt_file))

class ValueNetwork(nn.Module):
    '''Just estimates the value of a state or set of states,
    doesnt care about what action you took or are taking'''
    def __init__(self, beta, input_dims, fc1_dims=256,fc2_dims = 256,
                name='Value Network',chkpt_dir='C:\\Users\\wilian\\Desktop\\Booker\\RL'):
        super(ValueNetwork,self).__init__()
        self.beta = beta
        self.input_dims= input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, name +'_sac')
        
        self.fc1 = nn.Linear(*self.input_dims,self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    def forward(self,state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)
        v = self.v(state_value)
        return v
    
    def save_checkpoint(self):
        print('.....saving checkpoint.....')
        T.save(self.state_dict(), self.chkpt_file)
    def load_checkpoint(self):
        print('.......loading checkpoint........')
        self.load_state_dict(T.load(self.chkpt_file))
    

class ActorNetwork(nn.Module):
    '''Outputs a mean and stdev distribution of actions'''
    def __init__(self,alpha, input_dims,max_action,fc1_dims=256,fc2_dims = 256,
                n_actions = 2, name='Actor',chkpt_dir="C:\\Users\\wilian\\Desktop\\Booker\\RL"):
        '''max_action is the mulptiplication scalar so that the 
        actions in the environment are of the right scale'''
        super(ActorNetwork,self).__init__()
        self.alpha = alpha
        self.input_dims= input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.chkpt_dir = chkpt_dir
        self.n_actions =n_actions
        self.max_action = max_action
        self.reparam_noise = 1e-6 # can't take log of 0, 
        self.chkpt_file = os.path.join(self.chkpt_dir, name +'_sac')
        print(f"actor: {self.chkpt_file}")
        self.normal_tracker = []
        self.fc1 = nn.Linear(*self.input_dims,self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims,self.n_actions) #output is mean of probability distribution of each action for the policy
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)#output is stdev of probability distribution of each action for the policy
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    def forward(self,state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)
        
        mu = self.mu(prob)
        sigma = self.sigma(prob)
        # the clamp makes it so that the std dev isn't so wide.
        sigma = T.clamp(sigma,min = self.reparam_noise, max=1)
        
        return mu, sigma
    def sample_normal(self, state, reparameterize = True):
        mu, sigma = self.forward(state)
        self.normal_tracker.append((mu,sigma,state))
        #print('MU:',mu, 'sigma ', sigma)
        probabilites = Normal(mu, sigma)
        
        if reparameterize:
            actions = probabilites.rsample() #this gives a sample + some noise to encourage exploration
        else:
            actions = probabilites.sample()
        # tanh puts the value between -1 and 1, then multiply that by max value of action space to scale
        action_tanh = T.tanh(actions)
        action_scaled = T.tanh(actions) * T.tensor(self.max_action).to(self.device)
        log_probs = probabilites.log_prob(actions)  # for the loss function for updating network
        # .pow() is same as ** but tensor
        log_probs -= T.log(1-action_tanh.pow(2)+self.reparam_noise) # need the noise because log(0) = UNDEF
        log_probs = log_probs.sum(1,keepdim=True) # need a scalar quality to calculate loss 
        
        return action_scaled, log_probs
    
    
    def save_checkpoint(self):
        print('.....saving checkpoint.....')
        T.save(self.state_dict(), self.chkpt_file)
    def load_checkpoint(self):
        print('.......loading checkpoint........')
        self.load_state_dict(T.load(self.chkpt_file))


class SACAgent():
    '''Reward scale is for rewards and critic loss. takes into account 
    the entropy in system. Can be messed around with.
    
    tau is for doing the soft copy of the networks, instead of a 
    hard copy like in DQL
    '''
    def __init__(self,mem_cntr, alpha = 0.0003, beta = 0.0003, input_dims = [1],
                 env = None, gamma = 0.99,n_actions=2, max_size = 10000,tau = 0.005,
                layer1_size = 256, layer2_size = 256, batch_size = 200,
                reward_scale = 1):
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.input_dims = input_dims
        self.env = env
        self.gamma = gamma
        self.n_actions = n_actions
        self.max_size = max_size
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.batch_size = batch_size
        self.scale= reward_scale
        
        self.memory = ReplayBuffer(max_size, input_dims, n_actions = n_actions,mem_cntr = mem_cntr)
        #self.memory.save_memory()
        
        self.actor = ActorNetwork(self.alpha, self.input_dims,n_actions = n_actions,
                                  name = 'Actor', max_action = env.action_space.high)
        self.critic_1 = CriticNetwork(beta,input_dims,
                                      n_actions = n_actions, name = 'Critic_1')
        
        self.critic_2 = CriticNetwork(beta,input_dims,
                                      n_actions = n_actions, name = 'Critic_2')
        self.value = ValueNetwork(beta, input_dims,name = 'Value')
        self.target_value = ValueNetwork(beta,input_dims, 
                                         name = 'Target_value')
        
        self.update_network_parameters(tau=1) #does a hard copy of value network to target network on first time. Otherwise we will detune by tau
        
    def choose_action(self,state):
        ''' returns a numpy array of the mean of distribution for each action'''
        state = T.tensor([state],dtype=T.float).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize = False)
        
        return actions.cpu().detach().numpy()[0]
    def remember(self,state, action , reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
       
        
    def update_network_parameters(self, tau = None):
        '''updates the target_value network to be soft copy of value netork'''
        if tau is None: #this makes it so it does a soft copy for n=2 onward
            tau = self.tau
        
        target_value_params = self.target_value.named_parameters() #gets current params for target value network
        value_params = self.value.named_parameters() # gets current vals for params for value network
        
        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)
        
        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() +\
            (1-tau)*target_value_state_dict[name].clone()
        self.target_value.load_state_dict(value_state_dict)
    
    def save_models(self):
        print('.....saving models......')
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        #self.memory.save_memory()

            
    def load_models(self):
        print('.....saving models......')
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        #self.memory.load_memory()
    
    def learn(self,done):
        if self.memory.mem_cntr < self.batch_size:
            return
        #print(self.memory.mem_cntr,self.batch_size)
        state, action, reward, new_state, done = \
                    self.memory.sample_buffer(self.batch_size)
        rewards = T.tensor(reward,dtype=T.float).to(self.actor.device)
        dones = T.tensor(done).to(self.actor.device)
        new_states = T.tensor(new_state,dtype=T.float).to(self.actor.device)
        action = T.tensor(action,dtype=T.float).to(self.actor.device)
        states = T.tensor(state,dtype=T.float).to(self.actor.device)
        
        value = self.value(states).view(-1) #makes the datatype correct
        target_value = self.target_value(new_states).view(-1) #print this out without view and see what it does
        target_value[done] = 0.0 #???
        
        actions, log_probs = self.actor.sample_normal(states, reparameterize = False)
        log_probs = log_probs.view(-1)
        
        q1_new_policy = self.critic_1.forward(states,actions)
        q2_new_policy = self.critic_2.forward(states,actions)
        # take the min of the two q values. Stabilizes learning
        critic_value = T.min(q1_new_policy,q2_new_policy) 
        critic_value= critic_value.view(-1)
        
        self.value.optimizer.zero_grad()
        value_target = critic_value - 0.2*log_probs
        value_loss = 0.5* F.mse_loss(value,value_target)
        value_loss.backward(retain_graph =True)
        self.value.optimizer.step()
        
        actions, log_probs = self.actor.sample_normal(states, reparameterize = True) #reparameterize because we dont waant to lose gradient to update actor
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(states,actions)
        q2_new_policy = self.critic_2.forward(states,actions)
        #print(q1_new_policy,q2_new_policy, actions)
        # take the min of the two q values. Stabilizes learning
        critic_value = T.min(q1_new_policy,q2_new_policy) 
        critic_value= critic_value.view(-1)
        
        actor_loss = 0.2*log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph = True)
        self.actor.optimizer.step()
        
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        # q_hat includes entropy in loss function
        q_hat = self.scale*rewards + self.gamma*target_value
        q1_old_policy = self.critic_1.forward(states,action).view(-1)
        q2_old_policy = self.critic_2.forward(states,action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        
        self.update_network_parameters()



"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
Continuous version by Ian Danforth
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class ContinuousCartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 30.0
        self.tau = 0.02  # seconds between state updates
        self.min_action = -1.0
        self.max_action = 1.0

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,)
        )
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stepPhysics(self, force):
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
            (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        return (x, x_dot, theta, theta_dot)

    def step(self, action):
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))
        # Cast action to float to strip np trappings
        force = self.force_mag * float(action)
        self.state = self.stepPhysics(force)
        x, x_dot, theta, theta_dot = self.state
        done = x < -self.x_threshold \
            or x > self.x_threshold \
            or theta < -self.theta_threshold_radians \
            or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("""
You are calling 'step()' even though this environment has already returned
done = True. You should always call 'reset()' once you receive 'done = True'
Any further steps are undefined behavior.
                """)
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width /world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen-polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def close(self):
        if self.viewer:
            self.viewer.close()