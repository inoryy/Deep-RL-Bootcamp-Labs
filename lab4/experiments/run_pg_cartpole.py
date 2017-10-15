#!/usr/bin/env python

"""
This project was developed by Rocky Duan, Peter Chen, Pieter Abbeel for the Berkeley Deep RL Bootcamp, August 2017. Bootcamp website with slides and lecture videos: https://sites.google.com/view/deep-rl-bootcamp/.

Copyright 2017 Deep RL Bootcamp Organizers.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""


import chainer

from algs import pg
from env_makers import EnvMaker
from models import CategoricalMLPPolicy, MLPBaseline
from utils import SnapshotSaver
import numpy as np
import os
import logger

log_dir = "data/local/pg-cartpole"

np.random.seed(42)

# Clean up existing logs
os.system("rm -rf {}".format(log_dir))

with logger.session(log_dir):
    env_maker = EnvMaker('CartPole-v0')
    env = env_maker.make()
    policy = CategoricalMLPPolicy(observation_space=env.observation_space, action_space=env.action_space,
                                  env_spec=env.spec)
    baseline = MLPBaseline(observation_space=env.observation_space, action_space=env.action_space,
                           env_spec=env.spec)
    pg(
        env=env,
        env_maker=env_maker,
        n_envs=16,
        policy=policy,
        baseline=baseline,
        batch_size=2000,
        n_iters=100,
        snapshot_saver=SnapshotSaver(log_dir),
        optimizer=chainer.optimizers.Adam(1e-2)
    )
