"""
This project was developed by Rocky Duan, Peter Chen, Pieter Abbeel for the Berkeley Deep RL Bootcamp, August 2017. Bootcamp website with slides and lecture videos: https://sites.google.com/view/deep-rl-bootcamp/.

Copyright 2017 Deep RL Bootcamp Organizers.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""


from simplepg.simple_utils import register_test, nprs
import numpy as np
from chainer import Variable

register_test(
    "a2c.compute_returns_advantages",
    kwargs=lambda: dict(
        rewards=nprs(0).uniform(size=(5, 2)),
        dones=nprs(1).choice([True, False], size=(5, 2)),
        values=nprs(2).uniform(size=(5, 2)),
        next_values=nprs(3).uniform(size=(2,)),
        discount=0.99,
    ),
    desired_output=lambda: (
        np.array([[1.14554925, 1.25462372],
                  [0.60276338, 0.54488318],
                  [2.33579066, 1.90456042],
                  [1.93145037, 1.2713801],
                  [1.50895268, 0.38344152]]),
        np.array([[0.70955434, 1.22869749],
                  [0.0531009, 0.10956079],
                  [1.91542286, 1.5742256],
                  [1.72680173, 0.65210914],
                  [1.20929801, 0.11661424]])
    )
)

register_test(
    "a2c.compute_total_loss",
    kwargs=lambda: dict(
        logli=Variable(nprs(0).uniform(size=(10,)).astype(np.float32)),
        all_advs=Variable(nprs(1).uniform(size=(10,)).astype(np.float32)),
        ent_coeff=nprs(2).uniform(),
        ent=Variable(nprs(3).uniform(size=(10,)).astype(np.float32)),
        vf_loss_coeff=nprs(4).uniform(),
        all_returns=Variable(nprs(5).uniform(size=(10,)).astype(np.float32)),
        all_values=Variable(nprs(6).uniform(size=(10,)).astype(np.float32)),
    ),
    desired_output=lambda: (
        Variable(np.array(-0.4047563076019287, dtype=np.float32)),
        Variable(np.array(0.22883716225624084, dtype=np.float32)),
        Variable(np.array(-0.1834639459848404, dtype=np.float32))
    )
)
