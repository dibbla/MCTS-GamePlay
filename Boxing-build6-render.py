import gym
import math
from IPython import display
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

MAX_STEPS=20
MAX=10086
EXPLORATION=20

class Node:
    def __init__(self,parent):
        self.state=None
        self.done=False
        self.info=None
        self.t=0
        self.n=0
        self.parent=parent
        self.child=None #shall be a list
    
    def UCB1(self):
        global MAX, EXPLORATION
        if self.n==0:
            return MAX
        else:
            return self.t+2/(math.sqrt(2))*EXPLORATION*(math.sqrt(2*math.log(self.parent.n)/self.n))
        
def rollout(current,if_render):
    '''
    current is going to be a Node object
    '''
    sandBox=deepcopy(current.state)
    endReward=0
    done=False
    while done!=True:
        action=sandBox.action_space.sample()
        _,reward,done,info=sandBox.step(action)# wierd return obs_next
        endReward+=reward
    
        if if_render:
            img = plt.imshow(sandBox.render(mode='rgb_array'))
        
        if if_render:
            img.set_data(sandBox.render(mode='rgb_array')) # just update the data
            display.display(plt.gcf())
            display.clear_output(wait=True)
            action = sandBox.action_space.sample()
            sandBox.step(action)

    return endReward


def backPro(current,addnum):
    current.t+=addnum
    current.n+=1
    while current.parent!=None:
        current=current.parent
        current.t+=addnum
        current.n+=1
        
def MCTS(curr):
    # curr is a Node object
    
    if curr.child==None:
        curr.child=[]
        for i in range(curr.state.action_space.n):
            curr.child.append(Node(curr))
            curr.child[-1].state=curr.state
            curr.child[-1].state.step(i)
            
#     count=0        
    for itera in range(MAX_STEPS):
#         count+=1
#         print('round:',count,end=' ')
        
        #find UCB1 max child
        UCB1val=[]
        for child in curr.child:
            UCB1val.append(child.UCB1())
        UCBMaxChild=curr.child[UCB1val.index(max(UCB1val))]


        dis=UCBMaxChild
        while dis.child!=None:
            val=[]
            for i in dis.child:
                val.append(i.UCB1())
            dis=dis.child[val.index(max(val))]
        UCBMaxChild=dis
                
        if UCBMaxChild.n==0:
            reward=rollout(UCBMaxChild,False)
            backPro(UCBMaxChild,reward)
        
        #visited but no child, expand and rollout the first one
        elif UCBMaxChild.n>0 and UCBMaxChild.child==None:
            #expand
            UCBMaxChild.child=[]
            for i in range(UCBMaxChild.state.action_space.n):
                UCBMaxChild.child.append(Node(UCBMaxChild))
                UCBMaxChild.child[-1].state=UCBMaxChild.state
                UCBMaxChild.child[-1].state.step(i)
            #rollout the first one
            reward=rollout(UCBMaxChild.child[0],False)
            backPro(UCBMaxChild.child[0],reward)
        #######test#######
        val=[]
        for i in curr.child:
            val.append(i.UCB1())
#         print('UCB info:',val.index(max(val)),max(val))
#         print('UCBs:',val)

    result=[]
    for i in curr.child:
        result.append(i.t)
    return result.index(max(result))

env = gym.make('Breakout-ram-v0')
env.reset()
test=Node(None)
test.state=deepcopy(env)
last=0
#下面这些放在你需要渲染动画的地方，我放在了原来的env.render()的位置
img = plt.imshow(test.state.render(mode='rgb_array')) # only call this once
for _ in range(1000000):
    last+=1
    img.set_data(test.state.render(mode='rgb_array')) # just update the data
    display.display(plt.gcf())
    display.clear_output(wait=True)
    action = MCTS(test)
    _,reward,done,_=test.state.step(action)
#     print('one step')
#     re.append(reward)
    if done ==True:
        break
print('last:',last)
