# OpenGym CartPole-v0
# -------------------
#
# This code demonstrates use a full DQN implementation
# to solve OpenGym CartPole-v0 problem.
#
# Made as part of blog series Let's make a DQN, available at:
# https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
#
# author: Jaromir Janisch, 2016

import random, numpy, math, gym, sys
from keras.models import *
from keras.layers import *
from keras import backend as K

from tcl_env_dqn_1 import *
import tensorflow as tf

# ----------


DAY0 = 50
DAYN = 60
REWARDS = {}
for i in range(DAY0,DAYN,1):
    REWARDS[i]=[]

# -------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *




class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.model = self._createModel()
        self.model_ = self._createModel()

    def _createModel(self):
        l_input = Input(batch_shape=(None, self.stateCnt))
        l_input1 = Lambda(lambda x: x[:, 0:self.stateCnt - 7])(l_input)
        l_input2 = Lambda(lambda x: x[:, -7:])(l_input)
        l_input1 = Reshape((DEFAULT_NUM_TCLS, 1))(l_input1)
        l_Pool = AveragePooling1D(pool_size=self.stateCnt - 7)(l_input1)
        l_Pool = Reshape([1])(l_Pool)
        l_dense = Concatenate()([l_Pool, l_input2])
        l_dense = Dense(100, activation='relu')(l_dense)
        l_dense = Dropout(0.3)(l_dense)
        out_value = Dense(80, activation='linear')(l_dense)
        # model = Model(inputs=l_input, outputs=[out_tcl_actions,out_price_actions,out_deficiency_actions,out_excess_actions, out_value])
        model = Model(inputs=l_input, outputs=out_value)
        model._make_predict_function()
        opt = RMSprop(lr=0.00025)
        model.compile(loss='mse', optimizer=opt)
        return model

    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=100, epochs=epochs, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, self.stateCnt), target=target).flatten()

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())

    def downModel(self):
        self.model.set_weights(self.model_.get_weights())


# -------------------- MEMORY --------------------------
class Memory:  # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    def isFull(self):
        return len(self.samples) >= self.capacity


# -------------------- AGENT ---------------------------
MEMORY_CAPACITY = 500
BATCH_SIZE = 200

GAMMA = 1.0

MAX_EPSILON = 0.5
MIN_EPSILON = 0.004
LAMBDA = 1e-5 # speed of decay

UPDATE_TARGET_FREQUENCY = 200



class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)

    def act(self, s, deter):
        if deter == True:
            return numpy.argmax(self.brain.predictOne(s))
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt - 1)
        return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)
        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()
            self.brain.model.save_weights("DQNTNet.h5")
            print("Target model updated")
        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = self.epsilon = max(MAX_EPSILON -LAMBDA * self.steps, MIN_EPSILON)

    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)
        no_state = numpy.zeros(self.stateCnt)
        states = numpy.array([o[0] for o in batch])
        states_ = numpy.array([(no_state if o[3] is None else o[3]) for o in batch])
        p = self.brain.predict(states)
        p_ = self.brain.predict(states_, target=True)
        x = numpy.zeros((batchLen, self.stateCnt))
        y = numpy.zeros((batchLen, self.actionCnt))

        for i in range(batchLen):
            o = batch[i]
            s = o[0]
            a = o[1]
            r = o[2]
            s_ = o[3]

            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * numpy.amax(p_[i])
            x[i] = s
            y[i] = t
        self.brain.train(x, y)

class RandomAgent:
    memory = Memory(MEMORY_CAPACITY)

    def __init__(self, actionCnt):
        self.actionCnt = actionCnt

    def act(self, s, deter):
        return random.randint(0, self.actionCnt - 1)

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)


    def replay(self):
        pass


# -------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self,render= False):
        self.env = MicroGridEnv()
        self.render = render

    def run(self, agent,day=None):
        s = self.env.reset(day0=DAY0,dayn=DAYN,day=day)
        R = 0
        while True:
            # if self.render: self.env.render()
            a = agent.act(s,deter=self.render)

            s_, r, done, info = self.env.step(a)

            if done:  # terminal state
                s_ = None
            agent.observe((s, a, r, s_))
            if not self.render:
                agent.replay()
            s = s_
            R += r
            if done:
                # if self.render: self.env.render()
                break
        REWARDS[self.env.day].append(R)
        print("Total reward:", R)


# -------------------- MAIN ----------------------------
# PROBLEM = 'CartPole-v0'

env = Environment()
env1= Environment(render=True)
#
stateCnt = env.env.observation_space.shape[0]
actionCnt = env.env.action_space.n
#
agent = Agent(stateCnt, actionCnt)
randomAgent = RandomAgent(actionCnt)

# while randomAgent.memory.isFull() == False:
#     env.run(randomAgent)
#
# agent.memory.samples = randomAgent.memory.samples
# randomAgent = None
from time import time
import pickle


# t0= time()
# for _ in range(1000):
#     env.run(agent)
# print("Training finished")
# print("Training time: ",time()-t0)
#
# agent.brain.model.save_weights("DQNTNET.h5")
# with open("REWARDS_DQNTNET.pkl", 'wb') as f:
#     pickle.dump(REWARDS, f, pickle.HIGHEST_PROTOCOL)

# for rew in REWARDS.values():
#     # print(np.average(list(rew)))
#     pyplot.plot(list(rew))

# pyplot.legend(["Day {}".format(i) for i in range(11)], loc = 'upper right')
# pyplot.show()

agent.brain.model.load_weights("DQNTNET.h5")
env_test = Environment(render=True)
for day in range(DAY0,DAYN):
    env_test.run(agent,day=day)

print(np.average([list(REWARDS[i])[-1] for i in range(DAY0,DAYN)]))
with open("REWARDS_DQNTNET.pkl", 'wb') as f:
    pickle.dump(REWARDS, f, pickle.HIGHEST_PROTOCOL)
