
from osim.env import L2M2019Env
from sac import SACAgent
from osim.env import Arm2DVecEnv
import numpy as np
from sac import ContinuousCartPoleEnv

#env = Arm2DVecEnv(visualize=True)
#obs = env.reset()
#act =env.action_space.sample()
#obs =env.observation_space.sample()
#new_state, reward, done, info = env.step(act,obs_as_dict=False)
#print(new_state)
#print(f"act:{act}")
#print(f"act:{obs}")







#env = L2M2019Env(visualize=True,difficulty=1)
#env = Arm2DVecEnv(visualize=True)
env = ContinuousCartPoleEnv()

agent = SACAgent(mem_cntr=0,input_dims = env.observation_space.shape, env =env,
                n_actions = env.action_space.shape[0])




file_name = 'kfjdlkjf'
figure_file = 'plots/' +file_name
load_checkpoint = True
if load_checkpoint:
    agent.load_models()

score_history = []
n_games = 10
avg_scores = []


for i in range(n_games):

    done = False
    score =0
    #obs = env.reset(obs_as_dict=False) #for osim
    obs = env.reset()
    #print(obs)
    steps = 0
    while not done:
        #print(f"obs: {type(obs)}")
        act = agent.choose_action(obs)
        #new_state, reward, done, info = env.step(act, obs_as_dict=False) #for osim
        new_state, reward, done, info = env.step(act)
        env.render()
        #print(reward,score)
        score += reward
        obs = new_state
        steps+=1
        if steps == 1500:
            done = True
    print(score,steps)
    score_history.append(score)
    avg_score = sum(score_history[-100:])/100
    avg_scores.append(avg_score)
    """ if i == 0:
        agent = SACAgent(input_dims = env.observation_space.shape, env =env,
                n_actions = env.action_space.shape[0]) """

    print('episode ', i,'score: ', score,'avg score: ', avg_score)
