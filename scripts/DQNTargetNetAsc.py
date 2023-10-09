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

import random, numpy, math, gym, sys, threading
import threading
import time

from keras.models import *
from keras.layers import *
from keras import backend as K

from tcl_env_dqn import *
import tensorflow as tf

# ----------
# -- constants

RUN_TIME = 20
THREADS = 1
OPTIMIZERS = 1
THREAD_DELAY = 0.0001


HUBER_LOSS_DELTA = 1.0
LEARNING_RATE = 0.00025
REWARDS=[]

# ----------
def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2, L1)  # Keras does not cover where function in tensorflow :-(

    return K.mean(loss)


# -------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *


class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)
        self.default_graph = tf.get_default_graph()
        # self.model = self._build_model()
        # self.graph = self._build_graph(self.model)
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.model = self._createModel()
        self.model_ = self._createModel()
        print("models created")

        self.session.run(tf.global_variables_initializer())

        # self.default_graph.finalize()  # avoid modifications

    def _createModel(self):
        with self.default_graph.as_default():
            l_input = Input(batch_shape=(None, self.stateCnt))
            l_input1 = Lambda(lambda x: x[:, 0:self.stateCnt - 6])(l_input)
            l_input2 = Lambda(lambda x: x[:, -6:])(l_input)
            l_input1 = Reshape((DEFAULT_NUM_TCLS, 1))(l_input1)
            l_CNN1 = Conv1D(filters=3, kernel_size=7, activation='relu')(l_input1)
            # l_CNN2 = Conv1D(filters=3,kernel_size=3, activation='relu')(l_CNN1)
            l_Pool = AveragePooling1D(pool_size=3)(l_CNN1)
            l_Pool = Reshape([31 * 3])(l_Pool)
            l_dense1 = Dense(10, activation='relu')(l_Pool)
            l_dense1 = Dropout(0.1)(l_dense1)
            l_dense2 = Dense(60, activation='relu')(l_input2)
            l_dense2 = Dropout(0.1)(l_dense2)
            l_dense = Concatenate()([l_dense1, l_dense2])
            out_value = Dense(self.actionCnt, activation='linear')(l_dense)
            model = Model(inputs=l_input, outputs=out_value)
            model._make_predict_function()  # have to initialize before threading
            opt = RMSprop(lr=LEARNING_RATE)
            model.compile(loss=huber_loss, optimizer=opt)
            return model

    def train(self, x, y, epochs=50, verbose=0):
        with self.default_graph.as_default():
            self.model.fit(x, y, batch_size=64, epochs=epochs, verbose=verbose)

    def predict(self, s, target=False):
        with self.default_graph.as_default():
            if target:
                return self.model_.predict(s)
            else:
                return self.model.predict(s)

    def predictOne(self, s, target=False):
        with self.default_graph.as_default():
            return self.predict(s.reshape(1, self.stateCnt), target=target).flatten()

    def updateTargetModel(self):
        with self.default_graph.as_default():
            self.model_.set_weights(self.model.get_weights())


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
MEMORY_CAPACITY = 24000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001  # speed of decay

UPDATE_TARGET_FREQUENCY = 100


class Agent:

    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        # self.brain = Brain(stateCnt, actionCnt)
        # self.memory = Memory(MEMORY_CAPACITY)

    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt - 1)
        else:
            return numpy.argmax(brain.predictOne(s))

    def observe(self, sample, steps=None):  # in (s, a, r, s_) format
        memory.add(sample)
        if steps % UPDATE_TARGET_FREQUENCY == 0:
            brain.updateTargetModel()
            print("Target model updated")

        # slowly decrease Epsilon based on our eperience
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * steps)

    def replay(self):
        batch = memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([o[0] for o in batch])
        states_ = numpy.array([(no_state if o[3] is None else o[3]) for o in batch])

        p = brain.predict(states)
        p_ = brain.predict(states_, target=True)

        x = numpy.zeros((batchLen, self.stateCnt))
        y = numpy.zeros((batchLen, self.actionCnt))

        for i in range(batchLen):
            o = batch[i]
            s = o[0];
            a = o[1];
            r = o[2];
            s_ = o[3]

            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * numpy.amax(p_[i])

            x[i] = s
            y[i] = t

        brain.train(x, y)


class RandomAgent:
    memory = Memory(MEMORY_CAPACITY)

    def __init__(self, actionCnt):
        self.actionCnt = actionCnt

    def act(self, s):
        return random.randint(0, self.actionCnt - 1)

    def observe(self, sample,steps=None):  # in (s, a, r, s_) format
        memory.add(sample)

    def replay(self):
        pass


# -------------------- ENVIRONMENT ---------------------
class Environment(threading.Thread):
    stop_signal = False
    def __init__(self,agent=None,render= False):
        threading.Thread.__init__(self)
        self.env = MicroGridEnv()
        self.render = render
        if agent==None:
            self.agent=RandomAgent(self.env.action_space.shape[0])
        else:
            self.agent = agent

    def runEpisode(self):
        time.sleep(THREAD_DELAY)  # yield
        s = self.env.reset()
        R = 0
        while True:
            if self.render:
                self.env.render(s)

            a = self.agent.act(s)

            s_, r, done, info = self.env.step(a)

            if done:  # terminal state
                s_ = None

            self.agent.observe((s, a, r, s_),steps=len(REWARDS))
            self.agent.replay()

            s = s_
            R += r

            if done:
                if self.render:
                    self.env.render(s)
                break
        REWARDS.append(R)
        print("Total reward:", R)

    def run(self):
        while not self.stop_signal:
            self.runEpisode()

    def stop(self):
        self.stop_signal = True

# ---------
# class Optimizer(threading.Thread):
#     stop_signal = False
#
#     def __init__(self):
#         threading.Thread.__init__(self)
#
#     def run(self):
#         while not self.stop_signal:
#             agent.observe()
#
#     def stop(self):
#         self.stop_signal = True

# -------------------- MAIN ----------------------------
STEPS=0
env = Environment()
env1= Environment(render=True)

stateCnt = env.env.observation_space.shape[0]
actionCnt = env.env.action_space.shape[0]
memory = Memory(MEMORY_CAPACITY)
agent = Agent(stateCnt, actionCnt)
brain = Brain(stateCnt,actionCnt)
randomAgent = RandomAgent(actionCnt)

rand_envs = [Environment() for i in range(THREADS)]
for e in rand_envs:
    e.start()
time.sleep(RUN_TIME)
for e in rand_envs:
    e.stop()

for e in rand_envs:
    e.join()

randomAgent = None

REWARDS=[]

envs = [Environment(agent) for i in range(OPTIMIZERS)]
# for e in envs:
envs[0].start()
# time.sleep(RUN_TIME)
# for e in envs:
#     e.stop()
# for e in rand_envs:
#     e.join()

print("Training finished")
# finally:
#     agent.brain.model.save("cartpole-dqn.h5")


pyplot.close("all")
pyplot.plot(REWARDS)
pyplot.show()
