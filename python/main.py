#!/usr/bin/env python

import gym
import gym_sai2

import numpy as np
from data import *
from robotic_priors import *
from reinforcement_learning import DQN_Agent
from configs.dqn import config
from schedule import LinearSchedule
import threading

def wait():
    input("Hit <enter> to close.\n")
    exit()

if __name__ == "__main__":
    NUM_EPISODES = 10
    LEN_EPISODE  = 100
    NUM_ITERATIONS = 21

    # Create thread to kill process (ctrl-c doesn't work?)
    thread = threading.Thread(target=wait, daemon=True)
    thread.start()

    # Create sai2 environment
    env = gym.make("sai2-v0")
    # TODO: Implement environment seed
    env.seed(0)

    # State representation learning
    robotic_priors = RoboticPriors(env.dim_observation)
    robotic_priors.reset_session()
    dataLogDir=robotic_priors.create_logger() #get file for logger
    sess=robotic_priors.sess #get session

    # Reinforcement learning agent
    agent = DQN_Agent(env, sess, config, dataLogDir, logger=None)
    lrLine=LinearSchedule(config.lr_begin, config.lr_end, 20/2)
    epsLine=LinearSchedule(config.eps_begin, config.eps_end, 20/2)

    for i in range(NUM_ITERATIONS):

        print("====Main Iteration: {}====".format(i))
        # Generate RL trajectories
        with DataLogger() as d:
            filename = d.filename
            print("Simulate Arm")
            x_start=[0.00,-0.45,0.55]
            for j in range(NUM_EPISODES):

                print("\tEpisode: {}".format(j))

                # Get initial observation
                ob = env.reset()
                ob = np.array(ob)[np.newaxis,...]

                # One episode
                actions, observations, rewards, xs, dxs, learned_states, aInds, sp_hats, donemasks = [], [ob], [], [], [], [], [], [], []
                newRun=True
                # Generate trajectory
                for _ in range(LEN_EPISODE):
                    # Query RL policy
                    s_hat = robotic_priors.evaluate(np.reshape(ob, (1,-1)))
                    action, actionInd = agent.action(s_hat, epsLine.val)
                    
                    if newRun:
                        dxs.append(np.array(x_start)-np.array([0, -0.45, 0]))  #<-repurposing this field for privious state 
                    else:
                        dxs.append(np.array(info["x"])-np.array([0, -0.45, 0])) #<-repurposing this field for privious state 
                    ob, reward, done, info = env.step(action)
                    ob = np.array(ob)[np.newaxis,...]
                    sp_hat = robotic_priors.evaluate(np.reshape(ob, (1,-1)))

                    # Append to trajectory
                    #HACKY WAY TO GET IT TO TRAIN AGAINST x and xprime, THIS WILL BE SHITY CODE
                    actions.append(np.array(action)) # save index because one hot stuff
                    observations.append(ob)
                    rewards.append(reward)
                    xs.append(np.array(info["x"]))
                    #dxs.append(np.array(info["dx"]))   #SORRY HAVE TO DO THIS)                 
                    learned_states.append(s_hat)
                    donemasks.append(done)
                    sp_hats.append(sp_hat)
                    aInds.append(actionInd)

                    x_start=info["x"] #save for new start state
                
                #print(np.array(dxs))
                #print(np.array(xs))

                # Vectorize trajectory
                actions = np.row_stack(actions)
                observations = np.concatenate(observations, axis=0)
                rewards = np.array(rewards)
                xs = np.row_stack(xs)
                dxs = np.row_stack(dxs)
                learned_states = np.row_stack(learned_states)
                aInds= np.row_stack(aInds)
                sp_hats= np.row_stack(sp_hats)
                donemasks=np.row_stack(donemasks)
                #print(learned_states)
                # Log data
                d.log(j, actions, observations, rewards, xs, dxs, learned_states,aInds, sp_hats, donemasks) # Negin added aindex,.... 

            # Get list of actions, observations, rewards, xs, dxs, learned_states from all episodes
            episodes = d.flush()

        # Train represnetation learning
        dataLogDir=robotic_priors.create_logger()
        robotic_priors_data_generator = batch_data(data=episodes, extra=True, flatten=True)
        RL_data_generator = batch_data(data=episodes, extra=True, flatten=True)
        robotic_priors.train_network(robotic_priors_data_generator)
        
        if (i>0) and (i%2==0):
            agent.train_network(RL_data_generator, lrLine.val)
            lrLine.update(i)
            epsLine.update(i)

     

    env.close()

