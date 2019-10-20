# ud-drlnd-p1-navigation
Training an agent with the help of DQN to navigate (and collect bananas!) in a large, square world.

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Environment 

![Trained Agent][image1]

The environment provided is a modified version of Unity - ML Agents BANANA COLLECTOR

### GOAL 
The goal of the agent is to pick/collect the yellow bananas eliminatong the blue bananas ! This projects intention is to get the environment solved when the average return for consecutive 100 episode is greater than 13

### REWARD 
 The agent receives +1 when it collects Yellow Banana and -1 for Blue Banana
- **`+1`** - Yellow banana
- **`-1 ** - Blue Banana

### Action Space
 
The agent learns the best actions from the experience obtained from the observation
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.


### Getting Started

You can download the binaries Unity Modified version Environment that matches your Operating System:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    
 After downloading the file , unzip it at the root of this project 
 
    ```
    $ cd /path/to/udacity-drlnd-p1-navigation
    $ curl -LO https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip
    $ unzip Banana_Linux_NoVis.zip
    $ ls -l Banana_Linux_NoVis
    ```
    
# Training the agent : 

Let me demonstrate how to train the agent Using Jupyter notebook in Report.ipynb file 

To start the notebook server enter the following commands in your terminal

```bash
(base)$ conda activate drlnd
(drlnd)$ jupyter notebook
[I 07:12:35.811 NotebookApp] Serving notebooks from local directory: /home/thiyagarajan/jupyter
```

It will open up the web Browser . If its not automatically opening up the browser you manually copy the below line and paste into any of the browser

``` http://localhost:8888/?token=e406adf4c029503d5eae4888ec3bc9eb8b7f9a653e9d06bd ```
If you run the Report.ipynb Notebook the environment gets solved by 1000 episodes at the maximum possible case 


### Learning Algorithm
The agent is trained with Deep-Q-Learning Algorithm along with following hyperparametrs

* Neural Network Architecture:

|     Layers    | Config                       |
| ------------- | ------------------------     |
| Input layer   | 37 (observational space)     |
| Hidden Layer  | 1 ( Linear with 256 ) + ReLu |
| Output Layer  | Linear with 4 nodes          |



* Optimization Algorithm
  * Adam
    - Learning rate :
      * 5e-4


| Influencing factors  | rates |
|----------------------|-------|
| eps_start            | 1.0   |
| eps_end              | 0.01  |
| eps_decay            | 0.9   |
| BUffer_Size          | 1e5   |
| Batch_size           | 64    |
| Gamma                | 0.99  |
| TAU                  | 1e-3  |

### Results obtained :
Trained DQN parameters is saved in the file `qnetwork_local_checkpoint.pth` , The agent ables to solves the environment and task gets over at epsisode 288 ( recorded time) 

### Ideas for future work :

* Dueling Network : By implementing Dueling Network it can able to learn state space efficiently and to test out how the additional information about the compositon of action-values are added during estimation 

* Priotorized Experience Replay
