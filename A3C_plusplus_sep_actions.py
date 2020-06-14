# A3C++ a modified version of Asynchronous Advantage actor critic algorithm
# To solve the problem of migrogrid's energy management
# -----------------------------------
#
# A3C paper: https://arxiv.org/abs/1602.01783
#
# The A3C implementation is available at:
# https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/
# by: Jaromir Janisch, 2017

# Two variations are implemented: A memory replay and a deterministic search following argmax(pi) instead of pi as a probability distribution
# Every action selection is made following the action with the highest probability pi

# Author: Taha Nakabi
import os
import sys

import numpy as np
import tensorflow as tf

from matplotlib import pyplot
import gym, time, random, threading

from keras.models import *
from keras.layers import *
from keras import backend as K

from tcl_env_dqn_1 import *



MODELS_DIRECTORY = 'success'

# -- constants
RUN_TIME = 1000
THREADS = 16
OPTIMIZERS = 2
THREAD_DELAY = 0.000001




N_STEP_RETURN = 24
GAMMA = 1.0
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = .5
EPS_STOP = .001
EPS_STEPS = RUN_TIME*THREADS

MIN_BATCH = 200
TR_FREQ = 100
LEARNING_RATE = 1e-3

LOSS_V = 0.09  # v loss coefficient
LOSS_ENTROPY = 0.3 # entropy coefficient

# REWARDS = {}
# for i in range(DAY0,DAYN):
#     REWARDS[i]=[]

TRAINING_ITERATIONS = 1

# ---------
class Brain:
    train_queue = [[], [], [], [], []]  # s, a, r, s', s' terminal mask
    train_queue_copy = [[], [], [], [], []]  # s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()

    def __init__(self):
        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.model = self._build_model()
        self.graph = self._build_graph(self.model)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()
        self.rewards = {}
        for i in range(DAY0, DAYN):
            self.rewards[i]=0.0

        # self.default_graph.finalize()  # avoid modifications

    def _build_model(self):
        l_input = Input(batch_shape=(None, NUM_STATE))
        l_input1 = Lambda(lambda x: x[:, 0:NUM_STATE - 7])(l_input)
        l_input2 = Lambda(lambda x: x[:, -7:])(l_input)
        l_input1 = Reshape((DEFAULT_NUM_TCLS, 1))(l_input1)
        l_Pool = AveragePooling1D(pool_size=NUM_STATE - 7)(l_input1)
        l_Pool = Reshape([1])(l_Pool)
        l_dense = Concatenate()([l_Pool, l_input2])
        l_dense = Dense(100, activation='relu')(l_dense)
        l_dense = Dropout(0.3)(l_dense)
        out_tcl_actions= Dense(NUM_ACTIONS_TCLs, activation='softmax')(l_dense)
        out_price_actions= Dense(NUM_ACTIONS_PRICES, activation='softmax')(l_dense)
        out_deficiency_actions= Dense(NUM_ACTIONS_DEF, activation='softmax')(l_dense)
        out_excess_actions= Dense(NUM_ACTIONS_EXCESS, activation='softmax')(l_dense)
        # out = Dense(NUM_ACTIONS, activation='softmax')(l_dense)
        out_value = Dense(1, activation='linear')(l_dense)
        # model = Model(inputs=l_input, outputs=[out, out_value])
        model = Model(inputs=l_input, outputs=[out_tcl_actions,out_price_actions,out_deficiency_actions,out_excess_actions, out_value])
        model._make_predict_function()  # have to initialize before threading
        return model

    def _build_graph(self, model):
        s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATE))
        a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        r_t = tf.placeholder(tf.float32, shape=(None, 1))  # not immediate, but discounted n step reward
        tcl_p, price_p, deficiency_p, excess_p, v = model(s_t)

        a_t_tcl, a_t_price, a_t_def, a_t_excess = tf.split(a_t, [NUM_ACTIONS_TCLs, NUM_ACTIONS_PRICES, NUM_ACTIONS_DEF, NUM_ACTIONS_EXCESS], 1)
        log_prob_tcl = tf.math.log(tf.reduce_sum(input_tensor=tcl_p * a_t_tcl, axis=1, keepdims=True) + 1e-10)
        log_prob_price = tf.math.log(tf.reduce_sum(input_tensor=price_p * a_t_price, axis=1, keepdims=True) + 1e-10)
        log_prob_deficiency = tf.math.log(tf.reduce_sum(input_tensor=deficiency_p * a_t_def, axis=1, keepdims=True) + 1e-10)
        log_prob_excess = tf.math.log(tf.reduce_sum(input_tensor=excess_p * a_t_excess, axis=1, keepdims=True) + 1e-10)

        log_prob = log_prob_tcl + log_prob_price + log_prob_deficiency + log_prob_excess

        advantage = r_t - v
        loss_policy =  -log_prob * tf.stop_gradient(advantage)  # maximize policy
        loss_value = LOSS_V * tf.square(advantage)  # minimize value error

        entropy = LOSS_ENTROPY * (
                    tf.reduce_sum(input_tensor=tcl_p * tf.math.log(tcl_p + 1e-10), axis=1, keepdims=True) +
                    tf.reduce_sum(input_tensor=price_p * tf.math.log(price_p + 1e-10), axis=1, keepdims=True) +
                    tf.reduce_sum(input_tensor=deficiency_p * tf.math.log(deficiency_p + 1e-10), axis=1, keepdims=True) +
                    tf.reduce_sum(input_tensor=excess_p * tf.math.log(excess_p + 1e-10), axis=1, keepdims=True))  # maximize entropy (regularization)
        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)
        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE)
        minimize = optimizer.minimize(loss_total)
        return s_t, a_t, r_t, minimize, loss_total

    def optimize(self):
        if len(self.train_queue_copy[0])<TR_FREQ or len(self.train_queue_copy[0])<MIN_BATCH :
            time.sleep(0)  # yield
            return

        with self.lock_queue:
            if len(self.train_queue_copy[0])<TR_FREQ:  # more thread could have passed without lock
                return  # we can't yield inside lock
            self.train_queue = random.sample(np.array(self.train_queue).T.tolist(), MIN_BATCH)
            self.train_queue = np.array(self.train_queue).T.tolist()
            s, a, r, s_, s_mask = self.train_queue
            self.train_queue_copy = [[], [], [], [], []]

            s = np.vstack(s)
            a = np.vstack(a)
            r = np.vstack(r)
            s_ = np.vstack(s_)
            s_mask = np.vstack(s_mask)

        if len(s) > 5 * MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))

        v = self.predict_v(s_)
        r = r + GAMMA_N * v * s_mask  # set v to 0 where s_ is terminal state

        s_t, a_t, r_t, minimize, loss = self.graph
        # self.new_max()
        print("Training...")
        for _ in range(TRAINING_ITERATIONS):
            self.session.run([minimize,loss], feed_dict={s_t: s, a_t: a, r_t: r})
        print("Done...")

    # def new_max(self):
    #     length = max([len(self.rewards[i]) for i in self.rewards.keys()])
    #     # print("--------" + str(length))
    #     if length>10:
    #         R = np.average([np.average(self.rewards[i]) for i in self.rewards.keys() if self.rewards[i]!=[]])
    #         print("-------- R= " + str(R))
    #         print("-------- max reward  " + str(self.max_reward))
    #         if R > self.max_reward:
    #             print('new max found:')
    #             print(R)
    #             print("-------------------------------------------------------------------------------------------------")
    #             brain.model.save("A3C+++" +str()+".h5")
    #             print("Model saved")
    #             self.max_reward = R
    #         for i in range(0,DAYN-DAY0):
    #             self.rewards[i] = []

    def train_push(self, s, a, r, s_):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            self.train_queue_copy[0].append(s)
            self.train_queue_copy[1].append(a)
            self.train_queue_copy[2].append(r)

            if s_ is None:
                self.train_queue[3].append(NONE_STATE)
                self.train_queue[4].append(0.)

                self.train_queue_copy[3].append(NONE_STATE)
                self.train_queue_copy[4].append(0.)
            else:
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)

                self.train_queue_copy[3].append(s_)
                self.train_queue_copy[4].append(1.)

    def predict(self, s):
        with self.default_graph.as_default():
            tcl_p, price_p, deficiency_p, excess_p, v = self.model.predict(s)
            return [tcl_p, price_p, deficiency_p, excess_p], v

    def predict_p(self, s):
        with self.default_graph.as_default():
            tcl_p, price_p, deficiency_p, excess_p, v = self.model.predict(s)
            return [tcl_p[0], price_p[0], deficiency_p[0], excess_p[0]]

    def predict_v(self, s):
        with self.default_graph.as_default():
            tcl_p, price_p, deficiency_p, excess_p, v = self.model.predict(s)
            return v

    def predict_p_vote(self, s):
        # Boost learning. Several versions of the successfull models are voting for the best action
        votes=[]
        for filename in os.listdir(MODELS_DIRECTORY):
            if filename.endswith(".h5"):
                with self.default_graph.as_default():
                    try:
                        self.model.load_weights(MODELS_DIRECTORY+"/"+filename)
                        tcl_p, price_p, deficiency_p, excess_p, v = self.model.predict(s)
                        votes.append([np.argmax(tcl_p),np.argmax(price_p),np.argmax(deficiency_p),np.argmax(excess_p)])
                    except:
                        print(filename+"didn't vote!")
                        pass
        boosted_p = np.average(np.array(votes),axis=0)
        return np.rint(boosted_p).astype(int)


# ---------
frames = 0

class Agent:
    def __init__(self, eps_start, eps_end, eps_steps):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps
        self.memory = []  # used for n_step return
        self.R = 0.

    def getEpsilon(self):
        if (frames >= self.eps_steps):
            return self.eps_end
        else:
            return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps  # linearly interpolate

    def act(self, s,render=False):
        eps = self.getEpsilon()
        global frames
        frames = frames + 1

        if random.random() < eps:
            p = [np.random.dirichlet(np.ones(NUM_ACTIONS_TCLs),size=1)[0],
                 np.random.dirichlet(np.ones(NUM_ACTIONS_PRICES),size=1)[0],
                 np.random.dirichlet(np.ones(NUM_ACTIONS_DEF),size=1)[0],
                 np.random.dirichlet(np.ones(NUM_ACTIONS_EXCESS),size=1)[0]]
        else:
            s = np.array([s])
            if render:
                a = brain.predict_p_vote(s)
                p = np.random.dirichlet(np.ones(NUM_ACTIONS), size=1)
                print(a)
                return list(a), p
            p = brain.predict_p(s)
        # a = np.random.choice(NUM_ACTIONS, p=p.reshape(NUM_ACTIONS,))
        a = np.array([np.argmax(p[0]),np.argmax(p[1]),np.argmax(p[2]),np.argmax(p[3])])
        return list(a), p

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
        self.env = MicroGridEnv(day0=DAY0,dayn=DAYN)
        self.agent = Agent(eps_start, eps_end, eps_steps)


    def runEpisode(self,day=None):
        s = self.env.reset(day0=DAY0,dayn=DAYN,day=day)
        R = 0
        while True:
            time.sleep(THREAD_DELAY)  # yield
            if self.render:
                self.env.render(name='A3C++')
            a, p = self.agent.act(s,self.render)

            s_, r, done, _ = self.env.step(a)

            if done:  # terminal state
                s_ = None

            if not self.render:
                self.agent.train(s, np.concatenate(p), r, s_)
            s = s_
            R += r
            if done:
                if self.render: self.env.render(name='A3C++')
                break
        print(R)
        if self.render:
            return
        if R > brain.rewards[self.env.day] and  self.agent.getEpsilon()<0.1:
            print('new max found: '+str(R))
            print("-------------------------------------------------------------------------------------------------")
            try:
                brain.model.save("success4/A3C+++" + str(self.env.day) + ".h5")
                print("Model saved")
            except:
                pass
            brain.rewards[self.env.day] = R



    def run(self):
        while not self.stop_signal:
            self.runEpisode()

    def stop(self):
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
DAY0=40
DAYN=70
env_test = Environment(render=True, eps_start=0., eps_end=0.)
NUM_STATE = env_test.env.observation_space.shape[0]
NUM_ACTIONS = env_test.env.action_space_sep.shape[0]
NUM_ACTIONS_TCLs = 4
NUM_ACTIONS_PRICES = 5
NUM_ACTIONS_DEF = 2
NUM_ACTIONS_EXCESS = 2

NONE_STATE = np.zeros(NUM_STATE)
brain = Brain()  # brain is global in A3C
# Training
# ##########################################################################################################
#
# # brain.model.load_weights("A3C+++.h5")
#

if str(sys.argv[1])=='train':
    envs = [Environment() for i in range(THREADS)]
    opts = [Optimizer() for i in range(OPTIMIZERS)]
    t0=time.time()

    for o in opts:
        o.start()

    for e in envs:
        e.start()

    time.sleep(RUN_TIME)

    for e in envs:
        e.stop()
    for e in envs:
        e.join()

    for o in opts:
        o.stop()
    for o in opts:
        o.join()
    brain.model.save("success5/A3C+++"  + ".h5")
    print("Training finished")
    print('training_time:', time.time()-t0)
#################################################################################################################################################
# Test
while True:
    print('Models directory:')
    MODELS_DIRECTORY= input()
    if os.path.exists(MODELS_DIRECTORY):
        print("Day: ")
        try:
            day= int(input())
            env_test.runEpisode(day)
        except NameError:
            print(NameError)
            break

# import pickle
# with open("REWARDS_A3C+++f.pkl",'wb') as f:
#     pickle.dump(REWARDS,f,pickle.HIGHEST_PROTOCOL)

# for rew in REWARDS.values():
#     print(np.average(list(rew)))
#     pyplot.plot(list(rew))
# pyplot.legend(["Day {}".format(i) for i in range(11)], loc = 'upper right')
# pyplot.show()





# print(np.average([list(REWARDS[i])[-1] for i in range(11)]))

# pyplot.plot(REWARDS)
# pyplot.show()

