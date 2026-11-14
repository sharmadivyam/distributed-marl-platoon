"""
Controllers that drive a platoon in the simulator:
  * fixed: the OVM car-following controller with the best fixed setting (0.5, 0.5)
  * ia2c / neurcomm: the trained policies from demo/runs (best validation checkpoints, as in
    the report), acting deterministically like in the evaluation
"""
import configparser
import glob
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import main  # noqa: E402  (init_agent)
from simulator.network import Network  # noqa: E402
from simulator.world import Lane, NetworkedCACC  # noqa: E402

DEFAULT_RUNS = os.path.join(REPO, 'demo', 'runs')
FIXED_SETTING = (0.5, 0.5)
RUN_DIRS = {'ia2c': 'ia2c_catchup', 'neurcomm': 'ma2c_nc_catchup'}
LABELS = {'fixed': 'Fixed controller (0.5, 0.5), no learning',
          'ia2c': 'IA2C, trained (best checkpoint)',
          'neurcomm': 'NeurComm, trained (best checkpoint)'}


def read_config(run_dir):
    paths = glob.glob(os.path.join(run_dir, 'data', '*.ini'))
    if not paths:
        raise FileNotFoundError('no config in %s/data (train it first, see demo/DEMO_GUIDE.md)' % run_dir)
    config = configparser.ConfigParser()
    config.read(paths[0])
    return config


class FixedController:
    """Every car uses the same fixed OVM setting. Needs only the beacon from the car in front."""

    def __init__(self, env):
        self.action = env.a_map.index(FIXED_SETTING)

    def reset(self):
        pass

    def act(self, obs, done, env):
        return np.full(env.n_agent, self.action)


class LearnedController:
    """A trained IA2C or NeurComm model, acting like Trainer.perform in test mode."""

    def __init__(self, model, uses_fingerprint):
        self.model = model
        self.uses_fingerprint = uses_fingerprint

    def reset(self):
        self.model.reset()

    def act(self, obs, done, env):
        if self.uses_fingerprint:
            policy = self.model.forward(obs, done, env.get_fingerprint())
        else:
            policy = self.model.forward(obs, done)
        env.update_fingerprint(policy)
        return np.array([int(np.argmax(pi)) for pi in policy])


def load_model(env, config, run_dir):
    model = main.init_agent(env, config['MODEL_CONFIG'], 0, 0)
    if model is None or not model.load(os.path.join(run_dir, 'model', 'best', '')):
        raise FileNotFoundError('no best checkpoint in %s/model/best' % run_dir)
    return model


def build_lane(kind, runs_dir=DEFAULT_RUNS):
    net = Network()
    if kind == 'fixed':
        # the physics settings are identical in every CACC config; take them from a trained run
        config = read_config(os.path.join(runs_dir, RUN_DIRS['neurcomm']))
        env = NetworkedCACC(config['ENV_CONFIG'], net, comm='beacon')
        return Lane(kind, LABELS[kind], env, FixedController(env), net)
    run_dir = os.path.join(runs_dir, RUN_DIRS[kind])
    config = read_config(run_dir)
    comm = 'state' if kind == 'ia2c' else 'neurcomm'
    env = NetworkedCACC(config['ENV_CONFIG'], net, comm=comm,
                        n_hidden=config['MODEL_CONFIG'].getint('num_lstm'))
    model = load_model(env, config, run_dir)
    return Lane(kind, LABELS[kind], env, LearnedController(model, kind == 'neurcomm'), net)
