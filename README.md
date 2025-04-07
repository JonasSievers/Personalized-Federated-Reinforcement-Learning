# Personalized-Federated-Reinforcement-Learning
TODO: Abstract and Image
## Project structure
This repsoitory consists of the following parts: 
- **data**: contains all data gathering scripts and datasets used in the project
- **outputs**: all experiments create a unique output folder with the config, log and tensorboard logs 
- **src**: includes all necessary files to run experiments
  - **config**: hydra config folder
  - **envs**: contains the reinforcement learning environment
  - **utils**: helper classes for data handling, model generation and model networks

## Install and Run the project
To run the project you can fork this repository and following the instructions: First create your own virtual environment:
```
conda create --name pfrl python=3.12.7

conda activate pfrl

pip install -r requirements.txt
```

## Config Setup

## Pipeline
```
python src/main.py mode=local
```
```
python src/main.py mode=fed
```
```
python src/main.py mode=per
```
```
python src/main.py -m mode=local,fed,per
```
