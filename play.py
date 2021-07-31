import gym
from agents import QAgent
import numpy as np
from env import CrawlingRobotEnv

##Todo: Build a robot that can learn to crawl 
env = CrawlingRobotEnv(render = True)
agent = QAgent(env,gamma=0.9)
current_state = env.reset()

total_reward = 0
i = 0
while i<900000:
    i = i+1 
    action = agent.choose_action(current_state)
    next_state, reward, done, info = env.step(action)

    agent.learn(current_state,action,reward,next_state)
    current_state = next_state
    total_reward += reward

    if i%5000 == 0:
        print (total_reward/i)
        if (total_reward/i) >1.3:
            break

env = CrawlingRobotEnv(render = True)
current_state = env.reset()
total_reward = 0
agent.eps = 0
while i<900000:
    i = i+1   
    action = agent.choose_action(current_state)
    next_state, reward, done, info = env.step(action)

    agent.learn(current_state,action,reward,next_state)
    current_state = next_state
    total_reward += reward

    if i%5000 == 0:
        print (total_reward/i)
        if (total_reward/i) >1.3:
            break