from sac import SACAgent
from osim.env import L2M2019Env
import numpy as np
from osim.env import Arm2DVecEnv
import random
from sac import ContinuousCartPoleEnv
import gc
import pandas as pd


def main():
    data_path = "D:\\opensimrlmemory\\memory.csv"
    column_names = ['state_memory','new_state_memory','action_memory','reward_memory','terminal_memory']
    file = pd.DataFrame(columns=column_names,dtype=object)
    file.to_csv(data_path,mode='w',index=False)
    
    #env = L2M2019Env(visualize=False,difficulty=1)
    env = Arm2DVecEnv(visualize=False)
    #env = ContinuousCartPoleEnv()
    mem_cntr = 0
    agent = SACAgent(mem_cntr = mem_cntr,input_dims = env.observation_space.shape, env =env,
                    n_actions = env.action_space.shape[0])
    random.seed(0)

    file_name = 'kfjdlkjf'
    figure_file = 'plots/' +file_name
    load_checkpoint = False

    best_score = -1000000
    score_history = []
    n_games = 500
    avg_scores = []
    if load_checkpoint:
        agent.load_models()
    else:
        agent.save_models()
    
    for i in range(n_games):
        done = False
        score =0
        agent = SACAgent(mem_cntr = mem_cntr,input_dims = env.observation_space.shape, env =env,
                    n_actions = env.action_space.shape[0])
        agent.load_models()
        obs = env.reset(obs_as_dict=False) # for osim env
        #obs = env.reset()
        steps = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act, obs_as_dict=False)# for osim env
            #new_state, reward, done, info = env.step(act)
            #print('cntr',mem_cntr,agent.memory.reward_memory[0])
            agent.remember(obs,act,reward,new_state,done)
            #print('cntr',mem_cntr,agent.memory.reward_memory[0])
            
            agent.learn(done)
            score += reward
            obs = new_state
            steps+=1
            mem_cntr += 1
            if steps == 1000:
                done = True
        agent.save_models()
        del agent
        gc.collect()

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        avg_scores.append(avg_score)
        
        #if i %10 == 0:
            #agent.save_models()

        if score>best_score:
            best_score = score
            #if not load_checkpoint:
                #agent.save_models()
        print('episode ', i,'score: ', score,'avg score: ', avg_score, 'best score:',best_score)


if __name__ == "__main__":
    main()