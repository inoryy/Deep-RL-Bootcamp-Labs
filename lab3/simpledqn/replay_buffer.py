"""
This project was developed by Rein Houthooft, Rocky Duan, Peter Chen, Pieter Abbeel for the Berkeley Deep RL Bootcamp, August 2017. Bootcamp website with slides and lecture videos: https://sites.google.com/view/deep-rl-bootcamp/.

Code adapted from OpenAI Baselines: https://github.com/openai/baselines

Copyright 2017 Deep RL Bootcamp Organizers.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""


import numpy as np
import random
import pickle


class ReplayBuffer(object):
    def __init__(self, max_size):
        """Simple replay buffer for storing sampled DQN (s, a, s', r) transitions as tuples.

        :param size: Maximum size of the replay buffer.
        """
        self._buffer = []
        self._max_size = max_size
        self._idx = 0

    def __len__(self):
        return len(self._buffer)

    def add(self, obs_t, act, rew, obs_tp1, done):
        """
        Add a new sample to the replay buffer.
        :param obs_t: observation at time t
        :param act:  action
        :param rew: reward
        :param obs_tp1: observation at time t+1
        :param done: termination signal (whether episode has finished or not)
        """
        data = (obs_t, act, rew, obs_tp1, done)
        if self._idx >= len(self._buffer):
            self._buffer.append(data)
        else:
            self._buffer[self._idx] = data
        self._idx = (self._idx + 1) % self._max_size

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._buffer[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of transition tuples.

        :param batch_size: Number of sampled transition tuples.
        :return: Tuple of transitions.
        """
        idxes = [random.randint(0, len(self._buffer) - 1)
                 for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def dump(self, file_path=None):
        """Dump the replay buffer into a file.
        """
        file = open(file_path, 'wb')
        pickle.dump(self._buffer, file, -1)
        file.close()

    def load(self, file_path=None):
        """Load the replay buffer from a file
        """
        file = open(file_path, 'rb')
        self._buffer = pickle.load(file)
        file.close()
