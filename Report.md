# Report

## Solution

This work has been developed following the Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments [paper](https://arxiv.org/pdf/1706.02275.pdf). This is an adapted version of the DDPG algorithm (usually seen as a continuous control action spaces version of Q-learning) that works as follows.

* it has two kinds of networks the Actor (that predicts the action in multidimensional continuous spaces) and the Critic or Q (that estimates the cumulative reward of the episode usually using Temporal Difference methods),
* the Actor has as input the (partially) observed state of the environment of each agent and predict the best action based on the state,
* the Critic gets as input the observations and actions of all agents, thus is able to predict better the Q value function, which is very useful during training,
* in test the Critic is not used, so only the partial information observed by each agent is used,
* there are two networks of each kind a local network (updated at each step using gradient descent) and a target network updated softly with the weights of the local networks. This is done to stabilize learning as it can become very unstable,
* during learning noise is added to the Actor output in order to be in a exploratory mode, which is removed on test mode,
* there are several agent that work in independent environment and each one adds its experience to a replay buffer that is shared by all agents and, 
* the (local) actor and critic networks are updated using different samples from the replay buffer.

On this implementation we chose the following strategy to solve the environment,
* the (local) actor and critic networks are updated 20 times in a row (one for each agent), using 20 different samples from the replay buffer,
* the (target) actor and critic networks are updated once every two updates and with a soft update,
* gradient clipping is used on the critic network to further stabilize learning, 
* the Critic network has an input layer of 48 units (two times the state size - one for each agent), hidden layers of 128, 128 + 4 (two times the action size) and 64 units, and an output layer of 1 unit,
* the Actor network has an input layer of 24 units (state size), three hidden layer of 128, 128 and 32 units, and an output layer of 2 units (action size),
* noise dampening, with a function of the average score, so that when the average score approaches the maximum score of 1, the noise amplitude reduces.

The following values of the hyperparameters are used:
* BUFFER_SIZE = 1e5
* BATCH_SIZE = 128
* GAMMA = 0.99
* TAU = 1e-3
* LR_ACTOR = 1e-5
* LR_CRITIC = 1e-4
* WEIGHT_DECAY_CR = 0
* WEIGHT_DECAY_AC = 0
* UPDATE_EVERY = 2

## Results

We have done the training in two stages:

1. Training during 1000 episodes and store weights. The average score goes from 0 to 0.3.
2. Training during 143 episodes and store final weights. The average score goes from 0.2 to 0.51 and the agent stops.

![alt text](https://github.com/pablobd/CollabDRL-TennisGame/blob/master/tennis_perf.png)

The reason to do this is that training was very time and resource consuming and a good strategy was to split the training in three stages. The agent configuration for the three stages remains identical. However, the buffer is cleaned at the beginning of the three stages and experience starts to be stored from scratch.


## Improvements

In the next release, we plan to add the following improvements:
* Batch normalization after each layer
* Prioritized learning to speed up training

