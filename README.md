# Littoral-Anti-Surface-Warfare-Environment-and-Multi-Agent-Reinforcement-Learning

A simplistic RL environment for littoral surface warfare

The project includes a game environment that maps an image of the northern Baltic Sea into a game grid of 100 x 100 squares. Red and blue sides aim to annihilate one another in this littoral naval warfare scenario.
The environment supports multi-agent reinforcement logic, with each ship being an individual agent but having a joint reward calculation. The reward is formed based on compiling a comprehensive picture of the opposing side's vessels, making reasonable movements in the area and on successful engagements. Own losses inflict a penalty.

Albeit the environment is discrete in nature, as movement is grid-based, the environment is continuous with an action space of 4. This solution enables the ships to move anywhere (given that it is feasible i.e. clear of rocks or land) within their speed range and 360 degrees around them. The radar transmission and silence are also selected as a continuous value that is then discretized, as well as the engagement salvos. 

The observation space is compiled so that opposing units in radar coverage are seen unless the radar signals are obstructed by littoral areas such as peninsulas and islands. If units use active radar transmission, these can be detected by the opposing side halfway across the game grid, and having an electronic bearing from more than one unit can result in a successful fix that can enable missile engagement. A radar contact always allows for an engagement.

The key to successful tactics is to utilize the littoral area to cover its own units while exposing the opponent units for engagements. The gif animation below is a visualization of a test run where the Blue side handily annihilates the Red side. 

![animation](https://github.com/valauri/Littoral-Naval-Warfare-MARL/assets/71026684/f849cb66-9924-4d77-9a4c-be758d3674eb)
