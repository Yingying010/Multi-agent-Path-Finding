# Multi-agent-Path-Finding

This repository implements autonomous navigation for single and multiple robots using the PyBullet simulation library.

The implemented algorithms include 
- A*
- Dijkstra
- RRT
- RRT*
- Reinforcement Learning (RL)

The `finalChallenge_Astar` folder contains the implementation of the final challenge task using the A* algorithm. The tasks are divided into two scenarios: one with a fetch point as an intermediate waypoint and another without a fetch point.

The `finalChallenge_RRT` folder contains the implementation of the final challenge task using the RRT algorithm.

The `finalChallenge_RRTstar` folder contains the implementation of the final challenge task using the RRT* algorithm.

The `finalChallenge_dijkstra` folder contains the implementation of the final challenge task using the dijkstra algorithm.

The `fRRT_RRTstart` folder contains the implementation of the final challenge task using the dijkstra algorithm.

The `RRT_RRTstar` file implements path planning for both continuous and discrete actions. This version is not integrated with the final challenge task.

The `RL` folder contains a trained model for navigating a maze environment. The training results can be reviewed by running the `test.py` file. However, the performance of the trained model is currently suboptimal.
