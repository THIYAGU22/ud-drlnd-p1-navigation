# ud-drlnd-p1-navigation

Training an agent to navigate (and collect bananas!) in a large, square world.

# PROJECT OVERVIEW:

This projects aims to train an agent to navigate (and collect bananas!) in a large, square world.

# Rules
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

# Specifications at the user end
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

# Unity Env Controls

0 | move forward.
1 | move backward.
2 | turn left.
3 | turn right.

# Spec of the project as Dev:
*DQN networks helps to gather and store all the actions in ReplayBuffer with respective to the current policy and it accumulates

*Updating Q network by calculating Mean Squared Error loss with respective to target Q value and current Q output

Model achieves an average score of 13.0 in 288 episodes!!!!

![training progress off Dqn Model](https://github.com/THIYAGU22/ud-drlnd-p1-navigation/blob/master/training_progress.png?raw=true)
