# Littoral-Anti-Surface-Warfare-Environment-and-Multi-Agent-Reinforcement-Learning

A simplistic RL environment for littoral surface warfare

The project includes a game environment that maps an image of the northern Baltic Sea into a game grid of 100 x 100 squares. Red and blue sides aim to annihilate one another in this littoral naval warfare scenario.
The environment supports multi-agent reinforcement logic, with each ship being an individual agent but having a joint reward calculation. The reward is formed based on compiling a comprehensive picture of the opposing side's vessels, making reasonable movements in the area and on successful engagements. Own losses inflict a penalty if a defensive tactic is selected.

While the environment is discrete in nature, the agents can be either discrete or continuous. In the continuous environment, the action space consists of 2*4 values for normal distribution mean and standard deviation, while the discrete version produces separate action values of sizes 50, 5 and 2, respectively. This solution enables the ships to move anywhere (given that it is feasible, i.e. clear of rocks or land) within their speed range and 360 degrees around them. The radar transmission and silence are also selected as a continuous value that is then discretized, as well as the engagement salvos. 

The observation space is compiled so that opposing units in radar coverage are seen unless the radar signals are obstructed by littoral areas such as peninsulas and islands. If units use active radar transmission, these can be detected by the opposing side halfway across the game grid, and having an electronic bearing from more than one unit can result in a successful fix that can enable missile engagement. A radar contact always allows for an engagement.

The key to successful tactics is to utilize the littoral area to cover its own units while exposing the opponent units for engagements. The gif animation below is a visualization of a test run where the Blue side handily annihilates the Red side. 

![animation](https://github.com/valauri/Littoral-Naval-Warfare-MARL/assets/71026684/f849cb66-9924-4d77-9a4c-be758d3674eb)

The agents (DDQN and MAPPO) are in respective files. The combatant-class includes surface combatants of different sizes while the landing ship class defines solely landing ships. The environment is in game.py file and the hyperparameters for features are mainly managed in config.json file. The environment is run by calling main.py with boolean arguments for [skip training], [load models] and [visualize first test run].

To train an agent, the parameters have to be set in configuration file. The purpose is to train the Blue side first. Initially, the "trained" parameter is set to false and other options are selected as desired. Then the call
"python main.py false false false"
initializes training. If save_models is set to True, the resulting models are saved, after which the training can be swapped to use the trained model against an initial opposing agent.
