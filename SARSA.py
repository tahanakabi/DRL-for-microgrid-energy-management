# OpenGym CartPole-v0
# -------------------
#
# This code demonstrates use of a basic Q-network (without target network)
# to solve OpenGym CartPole-v0 problem.
#
# Made as part of blog series Let's make a DQN, available at:
# https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
#
# author: Jaromir Janisch, 2016


# --- enable this to run on GPU
# import os
# os.environ['THEANO_FLAGS'] = "device=gpu,floatX=float32"

import random, numpy, math, gym

# -------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.models import *
from keras.layers import *
from keras import backend as K


REWARDS = {}
for i in range(11):
    REWARDS[i]=[]

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        # self.model.load_weights("cartpole-basic.h5")

    def _createModel(self):
        l_input = Input(batch_shape=(None, self.stateCnt))
        l_dense=Dense(100, activation='relu')(l_input)
        # l_dense = Dropout(0.3)(l_dense)
        out_value = Dense(self.actionCnt, activation='linear')(l_dense)
        model = Model(inputs=l_input, outputs=out_value)
        model._make_predict_function()
        opt = RMSprop(lr=0.00025)
        model.compile(loss='mse', optimizer=opt)
        return model


    def train(self, x, y, epoch=1, verbose=0):
        self.model.fit(x, y, batch_size=100, epochs=epoch, verbose=verbose)

    def predict(self, s):
        return self.model.predict(s)

    def predictOne(self, s):
        return self.predict(s.reshape(1, self.stateCnt)).flatten()


# -------------------- MEMORY --------------------------
class Memory:  # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)
        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def extra(self, next_a):
        if len(self.samples) > 1:
            self.samples[-2][-1] = next_a

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)


# -------------------- AGENT ---------------------------
MEMORY_CAPACITY = 500
BATCH_SIZE = 200

GAMMA = 0.99

MAX_EPSILON = 0.4
MIN_EPSILON = 0.001
LAMBDA = 0.0004  # speed of decay


class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)

    def act(self, s, deter):
        if deter==True:
            return numpy.argmax(self.brain.predictOne(s))
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt - 1)

        return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)
        self.memory.extra(sample[1])
        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self):
        if len(self.memory.samples)<2:
            return
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([o[0] for o in batch])
        states_ = numpy.array([(no_state if o[3] is None else o[3]) for o in batch])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)

        x = numpy.zeros((batchLen, self.stateCnt))
        y = numpy.zeros((batchLen, self.actionCnt))

        for i in range(batchLen):
            o = batch[i]
            s = o[0]
            a = o[1]
            r = o[2]
            s_ = o[3]
            a_ = o[4]

            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * p_[i][a_]

            x[i] = s
            y[i] = t

        self.brain.train(x, y)


# -------------------- ENVIRONMENT ---------------------
from tcl_env_dqn import *

class Environment:
    def __init__(self, render = False):
        self.env = MicroGridEnv()
        self.render=render


    def run(self, agent, day=None):
        s = self.env.reset(day=day)
        R = 0
        while True:

            if self.render: self.env.render('SARSA')

            a = agent.act(s, deter=self.render)

            s_, r, done, info = self.env.step(a)

            if done:  # terminal state
                s_ = None

            agent.observe([s, a, r, s_,None])


            s = s_
            R += r

            if done:
                if self.render: self.env.render('SARSA')
                else:
                    agent.replay()
                break

        # REWARDS[self.env.day].append(R)
        print("Total reward:", R)


# -------------------- MAIN ----------------------------
# PROBLEM = TCLEnv
env = Environment()

stateCnt = env.env.observation_space.shape[0]
actionCnt = env.env.action_space.n

agent = Agent(stateCnt, actionCnt)

import pickle
import time
# t0=time.time()
# for _ in range(1000):
#     env.run(agent)
# print('training_time:', time.time()-t0)
# agent.brain.model.save_weights("SARSA.h5")
# with open("REWARDS_SARSA.pkl",'wb') as f:
#     pickle.dump(REWARDS,f,pickle.HIGHEST_PROTOCOL)
# for rew in REWARDS.values():
#     # print(np.average(list(rew)))
#     pyplot.plot(list(rew))
# pyplot.legend(["Day {}".format(i) for i in range(11)], loc = 'upper right')
# pyplot.show()
agent.brain.model.load_weights("SARSA.h5")
env_test=Environment(render=True)
# for day in range(11):
env_test.run(agent,day=200)
# print(np.average([list(REWARDS[i])[-1] for i in range(11)]))
