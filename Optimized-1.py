import gym
import math
import time
from IPython import display
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt

%matplotlib inline

MAX_STEPS = 200
EXPLORATION = 18/math.sqrt(2)


class Node:
    def __init__(self, parent):
        self.state = None
        self.done = False
        self.info = None
        self.t = 0
        self.n = 0
        self.parent = parent
        self.steps = []
        self.child = None  # shall be a list

    def UCB1(self):
        if self.n == 0:
            return 10086
        else:
            return self.t+2/(math.sqrt(2))*(30/math.sqrt(2)) * \
                (math.sqrt(2*math.log(self.parent.n)/self.n))


def rollout(node, if_render):
    enviro = gym.make('Breakout-ram-v0')
    enviro.reset()
    for i in node.steps:
        enviro.step(i)
    endReward = 0
    done = False
    while done != True:
        action = enviro.action_space.sample()
        _, reward, done, info = enviro.step(action)  # wierd return obs_next
        if reward>0:
            reward+=0.5
        endReward += reward-0.0005
    return endReward


def backPro(current, addnum):
    current.t += addnum
    current.n += 1
    while current.parent is not None:
        current = current.parent
        current.t += addnum
        current.n += 1


def MCTS(curr):
    # curr is a Node object

    if curr.child is None:
        curr.child = []
        for i in range(curr.state.action_space.n):
            curr.child.append(Node(curr))
            curr.child[-1].steps.append(i)

    for itera in range(100):
        start = time.perf_counter()
        # find UCB1 max child
        UCB1val = []
        for child in curr.child:
            UCB1val.append(child.UCB1())
        UCBMaxChild = curr.child[UCB1val.index(max(UCB1val))]

        dis = UCBMaxChild

        while dis.child is not None:
            val = []
            for i in dis.child:
                val.append(i.UCB1())
            dis = dis.child[val.index(max(val))]
        UCBMaxChild = dis

        if UCBMaxChild.n == 0:

            reward = rollout(UCBMaxChild, False)
            backPro(UCBMaxChild, reward)
            end = time.perf_counter()
            print('Iteration time:', end-start)
            continue

        # visited leaf node, expand and rollout the first one
        elif UCBMaxChild.n > 0 and UCBMaxChild.child is None:

            # expand
            UCBMaxChild.child = []
            for i in range(curr.state.action_space.n):
                UCBMaxChild.child.append(Node(UCBMaxChild))
                UCBMaxChild.child[-1].steps.append(i)
            end = time.perf_counter()
            # rollout the first one
            reward = rollout(UCBMaxChild.child[0], False)
            backPro(UCBMaxChild.child[0], reward)
            end = time.perf_counter()
            print('Iteration time:', end-start)
            continue



        end = time.perf_counter()
        print('Iteration time:', end-start)

    result = []
    for i in curr.child:
        result.append(i.t)
    print('result:',result)
    return result.index(max(result))


#############Main#################


env = gym.make('Breakout-ram-v0')
env.reset()
test = Node(None)
test.state = deepcopy(env)
test.state.seed(1995)
last = 0
print('O')
for _ in range(1000):
    test.child=None
    last += 1
    action = MCTS(test)
    _, reward, done, _ = test.state.step(action)
    print('\none step', 'reward:', reward, 'acton:', action, '\n')

    if done:
        break
print('last:', last)
