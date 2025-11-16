This folder contains the example code to reproduce the workflow presented in the paper:
"**Rethinking Explanation Evaluation under the Retraining Scheme**"

The results are presented directly in the three jupyter notebooks, each representing different part of the evaluation process, namely:
- `model_training.ipynb`: Model training to setup test environment
- `model_explaining.ipynb`: Explanation derivation on the trained model for further dataset manipulation
- `model_retraining.ipynb`: Retraining of the trained model for explanation evaluation

The cell outputs embedded in the notebookds correspond to test results acquired on the MNIST-CNN setting.
Changing the variable `DS_NAME` in the first cell of each notebook to `CIFAR10` will repeat the experiment on the CIFAR10-WRN40-4 setting.
Set `DEFAULT_DEVICE` to `cpu` if running the experiments in a non-GPU setting.

To reproduce the results, simply executing cells of the three notebooks in the listed order. 
Datasets required by the process will be downloaded automatically to `./data/{DS_NAME}`.
The other intermediate results created during the whole process, including trained models and derived explanations, will be put under the directories `./models` and `./explanations`, respectively.
The total running time can take several minutes depending on local hardware setting.


## Dependencies
- Python: 3.11.2
- numpy: 1.24.2
- timm: 1.0.15
- torch: 2.6.0
- torchvision: 0.21.0
- tqdm: 4.64.1
- CUDA: 12.5

## Download resources
The datasets required to reproduce the results will be downloaded and processed automatically when executing the notebook cells in sequential order.

## Overall Code structure
```
.
├── configs	# Training config for the selected test case
│   ├── cifar10_wrn40_4_config.yaml
│   └── mnist_cnn_config.yaml
├── data, explanations, models	# Storage of data and intermediate results, will be created automatically during evaluation
├── model_explaining.ipynb	# Notebook for explanation derivaiton
├── model_retraining.ipynb	# Notebook for explanation evaluation by retraining
├── model_training.ipynb	# Notebook for model training to setup test case
├── requirements.txt		# Versions of the main dependencies
└── utils
    ├── basic.py		# Basic utility function about logging
    ├── data_process.py		# Data preprocessing and dataset manipulation
    ├── explainers.py		# Implementation of basic attribution methods
    ├── my_models.py		# Implementation of model structures
    ├── setting_dicts.py	# Configuration of different test cases
    └── train_engine.py		# Training and validation loops
```