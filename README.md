# Distributed MARL Platoon

Our major project (Anshika Pandey, Divyam Sharma), supervised by Dr. Archit Somani.

We take the paper **"Multi-agent Reinforcement Learning for Networked System Control"**
(Chu, Chinchali and Katti, ICLR 2020), which proposes NeurComm, and run it as a real distributed
system: every car of the platoon is its own MPI process and the cars only talk to their
neighbours through messages. Then we break the network on purpose (lost messages, late
messages, crashed processes) and see what happens.

The code is built on the authors' PyTorch implementation:
https://github.com/cts198859/deeprl_network (their original README is below).

## What we added

| Folder | What it does |
|---|---|
| `demo/` | reproduction runs of IA2C and NeurComm on CACC catch-up, evaluation on 50 unseen seeds, reports |
| `simulator/` | browser simulator: watch the platoon live or replay an MPI run, with every V2V message drawn |
| `mpi/platoon_mpi.py` | the platoon on MPI: 1 road process + 1 process per car. Controllers: no control, radar only, fixed, IA2C, NeurComm |
| `mpi/check_mpi.py` | proves the MPI run equals the Python run at every step |
| `mpi/experiments.py` | message loss and delay sweeps (10 seeds), scaling with 8/16/32 cars |
| `mpi/train_mpi.py` | distributed training: each car learns in its own process, optional neighbour weight averaging |
| `mpi/check_train.py` | proves distributed training equals the original central training (up to float rounding) |
| `mpi/two_laptops/` | run the platoon across two laptops |

See `mpi/README.md` for results and all options.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r demo/requirements.txt mpi4py   # tested with Python 3.9
brew install open-mpi
# 9 processes: the road + 8 cars, each car runs its own part of the trained NeurComm network
mpirun --oversubscribe -np 9 .venv/bin/python mpi/platoon_mpi.py --controller neurcomm --log mpi/logs/nc.jsonl
.venv/bin/python mpi/check_mpi.py
bash mpi/run.sh                      # all demo runs, then the command to watch them in the browser
```

---

# Original README (Chu et al.)

# Networked Multi-agent RL (NMARL)
This repo implements the state-of-the-art MARL algorithms for networked system control, with observability and communication of each agent limited to its neighborhood. For fair comparison, all algorithms are applied to A2C agents, classified into two groups: IA2C contains non-communicative policies which utilize neighborhood information only, whereas MA2C contains communicative policies with certain communication protocols.

Available IA2C algorithms:
* PolicyInferring: [Lowe, Ryan, et al. "Multi-agent actor-critic for mixed cooperative-competitive environments." Advances in Neural Information Processing Systems, 2017.](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf)
* FingerPrint: [Foerster, Jakob, et al. "Stabilising experience replay for deep multi-agent reinforcement learning." arXiv preprint arXiv:1702.08887, 2017.](https://arxiv.org/pdf/1702.08887.pdf)
* ConsensusUpdate: [Zhang, Kaiqing, et al. "Fully decentralized multi-agent reinforcement learning with networked agents." arXiv preprint arXiv:1802.08757, 2018.](https://arxiv.org/pdf/1802.08757.pdf)


Available MA2C algorithms:
* DIAL: [Foerster, Jakob, et al. "Learning to communicate with deep multi-agent reinforcement learning." Advances in Neural Information Processing Systems. 2016.](http://papers.nips.cc/paper/6042-learning-to-communicate-with-deep-multi-agent-reinforcement-learning.pdf)
* CommNet: [Sukhbaatar, Sainbayar, et al. "Learning multiagent communication with backpropagation." Advances in Neural Information Processing Systems, 2016.](https://arxiv.org/pdf/1605.07736.pdf)
* NeurComm: Inspired from [Gilmer, Justin, et al. "Neural message passing for quantum chemistry." arXiv preprint arXiv:1704.01212, 2017.](https://arxiv.org/pdf/1704.01212.pdf)

Available NMARL scenarios:
* ATSC Grid: Adaptive traffic signal control in a synthetic traffic grid.
* ATSC Monaco: Adaptive traffic signal control in a real-world traffic network from Monaco city.
* CACC Catch-up: Cooperative adaptive cruise control for catching up the leadinig vehicle.
* CACC Slow-down: Cooperative adaptive cruise control for following the leading vehicle to slow down.

## Requirements
* Python3 == 3.5
* [PyTorch](https://pytorch.org/get-started/locally/) == 1.4.0
* [Tensorflow](http://www.tensorflow.org/install) == 2.1.0 (for tensorboard) 
* [SUMO](http://sumo.dlr.de/wiki/Installing) >= 1.1.0

## Usages
First define all hyperparameters (including algorithm and DNN structure) in a config file under `[config_dir]` ([examples](./config)), and create the base directory of each experiement `[base_dir]`. For ATSC Grid, please call [`build_file.py`](./envs/large_grid_data) to generate SUMO network files before training.

1. To train a new agent, run
~~~
python3 main.py --base-dir [base_dir] train --config-dir [config_dir]
~~~
Training config/data and the trained model will be output to `[base_dir]/data` and `[base_dir]/model`, respectively.

2. To access tensorboard during training, run
~~~
tensorboard --logdir=[base_dir]/log
~~~

3. To evaluate a trained agent, run
~~~
python3 main.py --base-dir [base_dir] evaluate --evaluation-seeds [seeds]
~~~
Evaluation data will be output to `[base_dir]/eva_data`. Make sure evaluation seeds are different from those used in training.    

4. To visualize the agent behavior in ATSC scenarios, run
~~~
python3 main.py --base-dir [base_dir] evaluate --evaluation-seeds [seed] --demo
~~~
It is recommended to use only one evaluation seed for the demo run. This will launch the SUMO GUI, and [`view.xml`](./envs/large_grid_data) can be applied to visualize queue length and intersectin delay in edge color and thickness. 

## Reproducibility
The paper results are based on an out-of-date SUMO version 0.32.0. We are re-running the experiments with SUMO 1.2.0 and will update the results soon. The pytorch impelmention is avaliable at branch [pytorch](https://github.com/cts198859/deeprl_network/tree/pytorch).

## Citation
For more implementation details and underlying reasonings, please check our paper [Multi-agent Reinforcement Learning for Networked System Control](https://openreview.net/forum?id=Syx7A3NFvH).
~~~
@inproceedings{
chu2020multiagent,
title={Multi-agent Reinforcement Learning for Networked System Control},
author={Tianshu Chu and Sandeep Chinchali and Sachin Katti},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=Syx7A3NFvH}
}
~~~


