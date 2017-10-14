"""
This project was developed by Rocky Duan, Peter Chen, Pieter Abbeel for the Berkeley Deep RL Bootcamp, August 2017. Bootcamp website with slides and lecture videos: https://sites.google.com/view/deep-rl-bootcamp/.

Copyright 2017 Deep RL Bootcamp Organizers.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""


from chainer import Variable

from simplepg.simple_utils import register_test, nprs
from utils import Gaussian
import numpy as np

register_test(
    "pg.compute_surr_loss",
    kwargs=lambda: dict(
        dists=Gaussian(
            means=Variable(nprs(0).uniform(size=(10, 3)).astype(np.float32)),
            log_stds=Variable(nprs(1).uniform(
                size=(10, 3)).astype(np.float32)),
        ),
        all_acts=Variable(nprs(2).uniform(size=(10, 3)).astype(np.float32)),
        all_advs=Variable(nprs(3).uniform(size=(10,)).astype(np.float32)),
    ),
    desired_output=lambda: Variable(
        np.array(1.9201269149780273, dtype=np.float32))
)
