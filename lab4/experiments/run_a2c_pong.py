#!/usr/bin/env python

"""
This project was developed by Rocky Duan, Peter Chen, Pieter Abbeel for the Berkeley Deep RL Bootcamp, August 2017. Bootcamp website with slides and lecture videos: https://sites.google.com/view/deep-rl-bootcamp/.

Copyright 2017 Deep RL Bootcamp Organizers.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""


from algs import a2c
from env_makers import EnvMaker
from models import CategoricalCNNPolicy
from utils import SnapshotSaver
import numpy as np
import os
import logger

log_dir = "data/local/a2c-pong"

np.random.seed(42)

# Clean up existing logs
os.system("rm -rf {}".format(log_dir))

with logger.session(log_dir):
    env_maker = EnvMaker('PongNoFrameskip-v4')
    env = env_maker.make()
    policy = CategoricalCNNPolicy(
        env.observation_space, env.action_space, env.spec)
    vf = policy.create_vf()
    a2c(
        env=env,
        env_maker=env_maker,
        n_envs=16,
        policy=policy,
        vf=vf,
        snapshot_saver=SnapshotSaver(log_dir, interval=10),
    )
