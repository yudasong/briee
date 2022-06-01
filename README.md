# Code for paper Provable Benefits of Representational Transfer in Reinforcement Learning
Paper link: [arXiv](https://arxiv.org/abs/2205.14571)

## Check out the code for Briee
[Code](https://github.com/yudasong/briee) for paper Efficient Reinforcement Learning in Block MDPs: A Model-free Representation Learning approach

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

To reproduce our result in comblock (Section 6.1), please run:
```bash
bash run.sh 
```

For online reptransfer, please run:
```bash
bash run_online.sh 
```

To reproduce our result in comblock with partitioned observation (Section 6.2), please run:
```bash
bash run_po.sh 
```

For online reptransfer, please run:
```bash
bash run_po_online.sh
```

To see all the hyperparameters, please refer to `utils.py`.
