# OpenGym CartPole-v0 with A3C on GPU
# -----------------------------------
#
# A3C implementation with GPU optimizer threads.
#
# Made as part of blog series Let's make an A3C, available at
# https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/
#
# author: Jaromir Janisch, 2017

import numpy as np
import tensorflow as tf

import gym, time, random, threading

from keras.models import *
from keras.layers import *
from keras import backend as K

from tcl_env_dqn import *
import pickle

# -- constants
RUN_TIME = 100000/(42*16)
THREADS = 16
OPTIMIZERS = 4
THREAD_DELAY = 0.00001

GAMMA = 0.9

N_STEP_RETURN = 24
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.4
EPS_STOP = .001
EPS_STEPS = 1000
PPO_EPS = 0.05
MIN_BATCH = 100
LEARNING_RATE = 1e-3

LOSS_V = 0.005  # v loss coefficient
LOSS_ENTROPY = 0.1  # entropy coefficient

REWARDS = {}
for i in range(11):
    REWARDS[i]=[]

TRAINING_ITERATIONS = 5

# ---------
class Brain:
    train_queue = [[], [], [], [], []]  # s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()

    def __init__(self):
        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.model = self._build_model()
        self.graph = self._build_graph(self.model)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()
        self.rewards = REWARDS.copy()

        # self.default_graph.finalize()  # avoid modifications

    def _build_model(self):
        l_input = Input(batch_shape=(None, NUM_STATE))
        l_dense = Dense(100, activation='relu')(l_input)
        # l_dense = Dropout(0.3)(l_dense)
        out= Dense(NUM_ACTIONS, activation='softmax')(l_dense)
        out_value = Dense(1, activation='linear')(l_dense)
        model = Model(inputs=l_input, outputs=[out, out_value])
        # model = Model(inputs=l_input, outputs=[out_tcl_actions,out_price_actions,out_deficiency_actions,out_excess_actions, out_value])
        model._make_predict_function()  # have to initialize before threading
        return model

    def _build_graph(self, model):
        s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATE))
        a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        r_t = tf.placeholder(tf.float32, shape=(None, 1))  # not immediate, but discounted n step reward
        old_log_p_t = tf.placeholder(tf.float32, shape=(None, 1))
        p, v = model(s_t)
        log_prob = tf.log(tf.reduce_sum(p * a_t, axis=1, keepdims=True) + 1e-10)
        ratio = tf.exp(log_prob - old_log_p_t)
        advantage = r_t - v
        surr1 = ratio * tf.stop_gradient(advantage)
        surr2 = tf.clip_by_value(ratio, 1.0 - PPO_EPS, 1.0 + PPO_EPS) * tf.stop_gradient(advantage)
        surr = tf.minimum(surr1, surr2)
        loss_policy = -surr  # maximize policy
        entropy = LOSS_ENTROPY * (tf.reduce_sum(p * tf.log(p + 1e-100), axis=1, keepdims=True))
        loss_value = LOSS_V * tf.square(advantage)  # minimize value error
        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)
        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE)
        minimize = optimizer.minimize(loss_total)
        return s_t, a_t, r_t, minimize, loss_total, old_log_p_t, log_prob

    def optimize(self):
        if len(self.train_queue[0]) < MIN_BATCH:
            time.sleep(0)  # yield
            return

        with self.lock_queue:
            if len(self.train_queue[0]) < MIN_BATCH:  # more thread could have passed without lock
                return  # we can't yield inside lock

            s, a, r, s_, s_mask = self.train_queue
            self.train_queue = [[], [], [], [], []]

        s = np.vstack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)

        if len(s) > 5 * MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))

        v = self.predict_v(s_)
        r = r + GAMMA_N * v * s_mask  # set v to 0 where s_ is terminal state

        s_t, a_t, r_t, minimize, loss, old_log_p_t, log_prob = self.graph
        print("Training")
        old_log_p = self.session.run(log_prob, feed_dict={s_t: s, a_t: a})
        for _ in range(TRAINING_ITERATIONS):
            self.session.run([minimize,loss], feed_dict={s_t: s, a_t: a, r_t: r,old_log_p_t:old_log_p})
        print("Done...")


    def train_push(self, s, a, r, s_):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if s_ is None:
                self.train_queue[3].append(NONE_STATE)
                self.train_queue[4].append(0.)
            else:
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)

    def predict(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p, v

    def predict_p(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p

    def predict_v(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return v


# ---------
frames = 0

class Agent:
    def __init__(self, eps_start, eps_end, eps_steps):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps
        self.random_action=False
        self.memory = []  # used for n_step return
        self.R = 0.

    def getEpsilon(self):
        if (frames >= self.eps_steps):
            return self.eps_end
        else:
            return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps  # linearly interpolate

    def act(self, s):
        eps = self.getEpsilon()
        global frames
        frames = frames + 1

        if random.random() < eps:
            p = np.random.dirichlet(np.ones(NUM_ACTIONS), size=1)
            self.random_action=True
        else:
            s = np.array([s])
            p = brain.predict_p(s)
            self.random_action=False
        a = np.random.choice(NUM_ACTIONS, p=p.reshape(NUM_ACTIONS,))
        # a = np.argmax(p.reshape(NUM_ACTIONS,))
        return a,p

    def train(self, s, a, r, s_):
        def get_sample(memory, n):
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n - 1]

            return s, a, self.R, s_

        a_cats = a
        # a_cats[a] = 1

        self.memory.append((s, a_cats, r, s_))

        self.R = (self.R + r * GAMMA_N) / GAMMA

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                brain.train_push(s, a, r, s_)

                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            brain.train_push(s, a, r, s_)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)


# possible edge case - if an episode ends in <N steps, the computation is incorrect

# ---------
class Environment(threading.Thread):
    stop_signal = False

    def __init__(self, render=False, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS):
        threading.Thread.__init__(self)

        self.render = render
        self.env = MicroGridEnv()
        self.agent = Agent(eps_start, eps_end, eps_steps)
        self.episode_counter=0


    def runEpisode(self,day=None):
        s = self.env.reset(day=day)
        R = 0

        while True:
            time.sleep(THREAD_DELAY)  # yield
            if self.render:
                # brain.model.load_weights("A3C-v=006_avg5.h5")
                self.env.render(name="PPO")
            a, p = self.agent.act(s)
            s_, r, done, _ = self.env.step(a)

            # if done:  # terminal state
            #     s_ = None
            aa=np.zeros(shape=(NUM_ACTIONS,))
            aa[a]=1
            self.agent.train(s, aa, r, s_)

            s = s_
            R += r
            if done:
                if self.render: self.env.render(name='PPO')
                break
        print(R)
        REWARDS[self.env.day].append(R)
        brain.rewards[self.env.day].append(R)




    def run(self):
        while not self.stop_signal:
            self.runEpisode()
            self.episode_counter += 1

    def stop(self):
        print(self.episode_counter)
        self.stop_signal = True


# ---------
class Optimizer(threading.Thread):
    stop_signal = False

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while not self.stop_signal:
            brain.optimize()

    def stop(self):
        self.stop_signal = True


# -- main
env_test = Environment(render=True, eps_start=0., eps_end=0.)
NUM_STATE = env_test.env.observation_space.shape[0]
NUM_ACTIONS = env_test.env.action_space.n
NUM_ACTIONS_TCLs = 4
NUM_ACTIONS_PRICES = 5
NUM_ACTIONS_DEF = 2
NUM_ACTIONS_EXCESS = 2

NONE_STATE = np.zeros(NUM_STATE)
EPISODE_COUNTER = 0
brain = Brain()  # brain is global in A3C

envs = [Environment() for i in range(THREADS)]
opts = [Optimizer() for i in range(OPTIMIZERS)]
# import time
# t0=time.time()
# for o in opts:
#     o.start()
#
# for e in envs:
#     e.start()
#
# time.sleep(RUN_TIME)
#
# for e in envs:
#     e.stop()
# for e in envs:
#     e.join()
#
# for o in opts:
#     o.stop()
# for o in opts:
#     o.join()
# AVGRWRD=[np.average(REWARDS[i:i+10]) for i in range(0,len(REWARDS),10)]
# print("Training finished")
# print('training_time:', time.time()-t0)
brain.model.load_weights("PPO_basic.h5")
# for rew in REWARDS.values():
#     pyplot.plot(list(rew))
# pyplot.legend(["Day {}".format(i) for i in range(11)], loc = 'upper right')
# pyplot.show()
# for day in range(11):
env_test.runEpisode(day=3)
# print(np.average([list(REWARDS[i])[-1] for i in range(11)]))
# with open("PPO_basic.pkl",'wb') as f:
#     pickle.dump(REWARDS,f,pickle.HIGHEST_PROTOCOL)


