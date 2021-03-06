# CollabDRL-TennisGame
Train two agents that control rackets to bounce a ball over a net.

## Environment Description
If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
* This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Technical Requirements

The project environment (provided by Udacity) is similar to, but not identical to the Tennis environment on the [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md). It is required to use this alternative project environment, which is available in this github page.

First of all, you need python 3 and conda. We suggest to use the [Anaconda distribution](https://www.anaconda.com/download/#linux), although other options are available. The project has the following dependencies: pytorch, matplotlib, numpy, and collections. You need to install these dependencies on a conda environment and create a kernel to run this notebook. Follow the instructions of the udacity DRL nano degree at github [repository drlnd](https://github.com/udacity/deep-reinforcement-learning) to create an environment for the project, install dependencies (pytorch, matplotlib, numpy, and collections) and create a kernel. You will not need to install Unity, because Udacity provides the Unity environment. Select the environment that matches your operating system:

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

## How it works

Open the file  jupyter notebook Tennis.ipynb and execute it reading carefully the instructions. Notice that when creating an environemnt for the game we forced the option no_graphics=True. You can change it to False to see a graphic representation of the game.
