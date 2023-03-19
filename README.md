# Deep Reinforcement Learning for Microgrid Energy Management
This repository contains an implementation of a Deep Reinforcement Learning (DRL) algorithm for managing the energy demand and supply of a microgrid. The implementation is built using Python and is based on the OpenAI Gym environment.

## Installation
To use this implementation, you will need to install the following dependencies:

Python 3.7 or higher
NumPy
TensorFlow
Keras
OpenAI Gym
Pandas
Matplotlib
To install these dependencies, you can use the following command:<br>

<code>pip install numpy tensorflow keras gym pandas matplotlib </code> <br>
## Usage
To train the DRL agent, you can use the main.py file. This file contains the main training loop and takes several command-line arguments, including the number of episodes to train for and the location of the output directory for saving the trained model.<br>

<code> main.py --episodes 1000 --output-dir models/ </code> <br>
To evaluate the performance of a trained model, you can use the evaluate.py file. This file takes the location of the saved model and the number of episodes to evaluate as command-line arguments.


<code> python evaluate.py --model models/model.h5 --episodes 100 </code> <br>
The repository also contains a set of pre-trained models, which can be used for quick evaluation without having to train the model yourself.

## Code Structure
The code is organized into several files:

**main.py**: Contains the main training loop for the DRL agent.<br>
**evaluate.py**: Contains code for evaluating a trained model.<br>
**agent.py**: Contains the implementation of the DRL agent.<br>
**environment.py**: Contains the implementation of the microgrid environment.<br>
**preprocess.py**: Contains code for preprocessing the input data.<br>
**visualize.py**: Contains code for visualizing the performance of the trained agent.<br>
**config.py**: Contains configuration parameters for the DRL agent.<br>
## Contributing
Contributions to this repository are welcome! If you find a bug or have an idea for an improvement, please submit a pull request.<br>

## License
This code is released under the MIT License. More information about this project can be found at: https://doi.org/10.1016/j.segan.2020.100413
