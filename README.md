# Deep Reinforcement Learning for Microgrid Energy Management
This repository contains an implementation of a Deep Reinforcement Learning (DRL) algorithm for managing the energy demand and supply of a microgrid. The implementation is built using Python and is based on the OpenAI Gym environment.

## Installation
Clone the repository and navigate to the directory <br>
Create a conda environment <br>
<code> conda env create -f conda.yaml </code> <br>
Activate the environment <br>
<code> conda activate tf2-gpu </code>
## Usage
To train the DRL agent, you can use the A3C_plusplus.py file. <br>
<code> python A3C_plusplus.py --train </code> <br>

To evaluate the performance of a trained model, you can use the same file with the option --test. <br>

<code> python A3C_plusplus.py --test </code> <br>

## Contributing
Contributions to this repository are welcome! If you find a bug or have an idea for an improvement, please submit a pull request.<br>

## License
This code is released under the MIT License. More information about this project can be found at: https://doi.org/10.1016/j.segan.2020.100413
