# SelfDrivingCarDRL
Building an autonomous driving agent using deep reinforcment learning.
Agent will try to learn to drive the car through AirSim Neghbourhood envivorment.
Double DQN will be used for agent to find the optimal policy.
For exploration Epsilon decreasing strategy is implemented.
And reward funtcion is crafted from the distance between car and center of the road, as well as car speed. 
Episode terminates when agent drives car into an obstacle, bumping into obstacle is punished with negative reward.
