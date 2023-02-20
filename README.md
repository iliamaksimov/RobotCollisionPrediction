# RobotCollisionPrediction
A Neural Network that helps a small robot navigate a simulated environment without any collisions. The NN is trained in PyTorch using Adam algorithm and predicts whether the robot is going to collide with any walls or objects in the next time step. This is a university coursework project.


<img width="1197" alt="Screen Shot 2023-02-20 at 9 49 39" src="https://user-images.githubusercontent.com/125837844/219986323-683e41c1-b6fe-4809-8d16-174d840b148e.png">

## What I Learned
* Collect and manage a dataset used to train and test a neural network for a robotics task.
* Prune collected data to balance out its distribution
* Design your own neural network architecture in PyTorch.
* Evaluate and improve a neural network model and verify an application in simulation.

## Instructions

### Python and Packages
* Python 3.7
* cython 0.29.32
* matplotlib 3.5.3
* scikit-learn 0.21.1
* scipy 1.7.3
* pymunk 5.7
* pillow 9.2.0
* pygame 2.1.2
* numpy 1.21.5
* noise 1.2.2

### Usage and Key Files description
* Run goal_seeking.py to start the simulation
* Networks.py contains the NN architecture 
* train_model.py is used for actually training the NN
* Data_Loaders.py contains PyTorch classes implementing train and test dataset iterables
