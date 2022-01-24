# Code for paper Efficient Reinforcement Learning in Block MDPs: A Model-free Representation Learning approach
Paper link: [In preparation]

## Prerequisites

Creating a virtual environment is recommended (or using conda alternatively):
```bash
pip install virtualenv
virtualenv /path/to/venv --python=python3

#To activate a virtualenv: 

. /path/to/venv/bin/activate
```

To install the dependencies (the results from the paper are obtain from gym==0.14.0):
``` bash
pip install -r requirements.txt
```

To install pytorch, please follow [PyTorch](http://pytorch.org/). Note that the current implementation does not require pytorch gpu.

We use [wandb](https://wandb.ai/home) to perform result collection, please setup wandb before running the code or add `os.environ['WANDB_MODE'] = 'offline'` in `main.py`.

## Run our code

To reproduce our result in comblock, please run:
```bash
bash run.sh [horizon] [num_threads] [save_path]
```

To reproduce our result in comblock with simplex feature, please run:
```bash
bash run_simplex.sh [num_threads] [save_path]
```

To reproduce our result in comblock with dense reward, please run:
```bash
bash run_dense.sh [num_threads] [save_path]
```

To see all the hyperparameters, please refer to `utils.py`.


## PPO+RND

Please refer to [this repo](https://github.com/yudasong/PCPG) for reproducing PPO+RND results.
