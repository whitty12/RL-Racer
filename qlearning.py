import numpy as np
import random
from collections import namedtuple
import matplotlib.pyplot as plt
import itertools


# q-learning algorithm
# Parameters:
#   alpha: (0, 1] step size
#   epsilon: small, > 0, for epsilon greedy
# Return:
#   action

class Q_Learner():

    def __init__(self, alpha, epsilon):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = .9
        self.action_set = [(0, 0, 0), (-.5, .1, 0), (.5, .1, 0), (0, 1, 0), (0, 0, .8)]
        # state set: [0,1,-1]x[-1,1]
        # self.state_set = [(0,-1), (1,-1), (-1,-1), (0,1), (1,1), (-1,1)]
        self.state_set = self.all_cropped_states()
        self.q = dict()
        self.init_q()
        self.curr_state = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
        self.curr_action = (0,0,0)
        self.a = 0

    def step(self, state):
        # choose A form S using policy derived from Q
        self.curr_action = self.get_action(self.curr_state)
        return self.curr_action

    def reward(self, state, action):
        if state is None or action is None:
            return 0
        # state = self.three_state(self.fix_state(state), action)
        state = self.cropped_state(state)
        # print(state)
        if action == [0,0,0]: return -1.0 # do nothing action
        if state[11] == 0: return -3.0    # off the track
        if state[10] == 0: return 1.0
        if abs(action[0]) > 0: return 1.0 #turning
        if action[1] > 0: return 1.0       # accelerating
        return 0

    def update(self, next_state, reward, done):
        self.a += 1
        # update Q
        # next_state = self.fix_state(next_state)
        # next_state = self.three_state(next_state, self.curr_action)
        next_state = self.cropped_state(next_state)
        old_value = self.q[(self.curr_state, self.curr_action)]
        _, next_max = self.max_action_reward(next_state)
        self.q[(self.curr_state, self.curr_action)] = (
            old_value + self.alpha * (reward + self.gamma * max(next_state) - old_value))
        self.curr_state = next_state

    def random_action(self):
        rand_action = self.action_set[random.randint(0,3)]
        # print("random action", rand_action)
        return rand_action

    def get_action(self, state):
        chance = random.uniform(0,1)
        if chance < self.epsilon:
            return self.random_action()
        else:
            max_action, _ = self.max_action_reward(state)
            # print("greedy action", max_action)
            return max_action

    def max_action_reward(self, state):
        max_action = (0,0,0)
        max_reward = 0
        for action in self.action_set:
            if self.q[(state, action)] > max_reward:
                max_action = action
                max_reward = self.q[(state, action)]
        return max_action, max_reward

    def init_q(self):
        for state in self.state_set:
            for action in self.action_set:
                self.q[(state, action)] = 0
        print(self.q)

    def three_state(self, orig_state, action):
        midpoint = int(len(orig_state) / 2)
        left = 0
        right = 0
        for i in range(len(orig_state)):
            for j in range(0, midpoint):
                if orig_state[i,j] == 255:
                    left += 1
            for j in range(midpoint, len(orig_state)):
                if orig_state[i,j] == 255:
                    right += 1

        # Boolean flag: 1 if on track, 0 not
        on_midpoint = (
                orig_state[int(len(orig_state)-2), int(len(orig_state)/2)] == 255 or
                orig_state[int(len(orig_state)-2), int(len(orig_state)/2)-1] == 255
        )

        s1 = 0
        s2 = -1

        if left > right:
            s1 = -1
        if right > left:
            s1 = 1
        if on_midpoint:
            s2 = 1

        new_state = (s1, s2)

        return new_state

    def fix_state(self, orig_state):
        new_state = np.zeros(np.shape(orig_state)[:-1], np.uint8)
        for i in range (len(orig_state)):
            for j in range (len(orig_state[i])):
                # if green, black, or white
                # print(orig_state[i,j,1])
                if (orig_state[i, j, 1] >= 200    # green channel is high
                    or orig_state[i, j, 1] == 0): # black
                    new_state[i, j] = 0
                else:   # greys - on the track
                    new_state[i, j] = 255
        return new_state

    def cropped_state(self, orig_state):
        orig_state = self.fix_state(orig_state)
        new_state = orig_state[len(orig_state)-3:len(orig_state)-1,
                    int(len(orig_state)/2-4): int(len(orig_state)/2+4)]
        new_state = tuple(np.reshape(new_state, 16))
        return new_state

    def all_cropped_states(self):
        states = []
        for state in itertools.product((255, 0), repeat=16):
            states.append(state)
        return states
