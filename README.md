# Littoral-Anti-Surface-Warfare-Environment-and-Multi-Agent-Reinforcement-Learning

A RL environment for littoral surface warfare

The project includes a game environment that maps an image of the northern Baltic Sea into a game grid of 100 x 100 squares. Red and blue sides aim to annihilate one another in this naval warfare scenario.
The logic utilizes multi-agent reinforcement logic, with each ship being an individual agent but having a joint reward calculation. The reward comprises of compiling a comprehensive picture of the opposing side's vessels, on successful engagements and with a penalty on losses on the own side.

There are altogether 28 actions for faster agents and 21 for slower units. The initial setup is that medium-sized ships are slower than small or large combatants. The actions are discrete and thus slower units have fewer actions as they cannot move as far as faster units in one turn. 

The observation space is gathered so that opposing units in radar coverage are seen unless the radar signals are obstructed by littoral areas such as peninsulas and islands. If units use active radar transmission, these can be detected by the opposing side by almost the full size of the game grid, and having an electronic bearing from more than one unit can result in a successful fix that can enable missile engagement. A radar contact always allows for an engagement.

The key to successful tactics is to utilize the littoral area to cover its own units while exposing the opponent units for engagements. If missiles are depleted, units are to move to predetermined replenishment points in order to carry on with engagements. 

