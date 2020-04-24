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
from matplotlib import pyplot
import gym, time, random, threading

from keras.models import *
from keras.layers import *
from keras import backend as K

from tcl_env import *

# -- constants

RUN_TIME = 100
THREADS = 16
OPTIMIZERS = 1
THREAD_DELAY = 0.01

GAMMA = 1.0

N_STEP_RETURN = 24
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.4
EPS_STOP = .05
EPS_STEPS = 7500

MIN_BATCH = 100
LEARNING_RATE = 5e-3

LOSS_V = 0.05  # v loss coefficient
LOSS_V_LIST = []
LOSS_ENTROPY = 1.5  # entropy coefficient

REWARDS = []
TRAINING_ITERATIONS = 250

# ---------
class Brain:
    train_queue = [[], [], [], [], []]  # s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()

    def __init__(self):
        self.session = tf.compat.v1.Session()
        # K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.model = self._build_model()
        self.graph = self._build_graph(self.model)

        self.session.run(tf.compat.v1.global_variables_initializer())
        self.default_graph = tf.compat.v1.get_default_graph()

        self.default_graph.finalize()  # avoid modifications

    def _build_model(self):

        l_input = Input(batch_shape=(None, NUM_STATE))
        l_input1 = Lambda(lambda x:x[:,0:NUM_STATE-7])(l_input)
        l_input2 = Lambda(lambda x:x[:,-7:])(l_input)
        l_input1 = Reshape((DEFAULT_NUM_TCLS, 1))(l_input1)
        # l_CNN1 = Conv1D(filters=3,kernel_size=7, activation='relu')(l_input1)
        # l_CNN2 = Conv1D(filters=3,kernel_size=3, activation='relu')(l_CNN1)
        l_Pool = AveragePooling1D(pool_size=3)(l_input1)
        l_Pool = Reshape([31*3])(l_Pool)
        l_dense1 = Dense(10, activation='relu')(l_Pool)
        l_dense1 = Dropout(0.1)(l_dense1)
        l_dense2 = Dense(60, activation='relu')(l_input2)
        l_dense2 = Dropout(0.1)(l_dense2)
        l_dense = Concatenate()([l_dense1,l_dense2])
        l_dense = Dropout(0.1)(l_dense)
        # l_dense = Dense(100, activation='relu')(l_dense)
        # l_dense = Dense(16, activation='relu')(l_dense)

        out_tcl_actions = Dense(NUM_ACTIONS_TCLs, activation='softmax')(l_dense)
        out_price_actions = Dense(NUM_ACTIONS_PRICES, activation='softmax')(l_dense)
        out_deficiency_actions = Dense(NUM_ACTIONS_DEF, activation='softmax')(l_dense)
        out_excess_actions = Dense(NUM_ACTIONS_EXCESS, activation='softmax')(l_dense)

        out_value = Dense(1, activation='linear')(l_dense)
        model = Model(inputs=l_input, outputs=[out_tcl_actions,out_price_actions,out_deficiency_actions,out_excess_actions, out_value])
        model._make_predict_function()  # have to initialize before threading
        return model

    def _build_graph(self, model):
        s_t = tf.compat.v1.placeholder(tf.float32, shape=(None, NUM_STATE))
        a_t = tf.compat.v1.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        r_t = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))  # not immediate, but discounted n step reward

        tcl_p, price_p, deficiency_p, excess_p, v = model(s_t)

        a_t_tcl, a_t_price, a_t_def, a_t_excess = tf.split(a_t, [4, 5, 2, 2], 1)
        # p = tf.concat([tcl_p, price_p, deficiency_p, excess_p], axis=1)
        # log_prob = tf.log(tf.reduce_sum(p * a_t, axis=1, keepdims=True) + 1e-10)

        log_prob_tcl = tf.math.log(tf.reduce_sum(input_tensor=tcl_p * a_t_tcl, axis=1, keepdims=True) + 1e-10)
        log_prob_price = tf.math.log(tf.reduce_sum(input_tensor=price_p * a_t_price, axis=1, keepdims=True) + 1e-10)
        log_prob_deficiency = tf.math.log(tf.reduce_sum(input_tensor=deficiency_p * a_t_def, axis=1, keepdims=True) + 1e-10)
        log_prob_excess = tf.math.log(tf.reduce_sum(input_tensor=excess_p * a_t_excess, axis=1, keepdims=True) + 1e-10)
        log_prob = log_prob_tcl + log_prob_price + log_prob_deficiency + log_prob_excess
        advantage = r_t - v

        loss_policy =  -log_prob * tf.stop_gradient(advantage)   # maximize policy
        loss_value = LOSS_V * tf.square(advantage)  # minimize value error

        # entropy = LOSS_ENTROPY * (tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keepdims=True))
        entropy = LOSS_ENTROPY * (tf.reduce_sum(input_tensor=tcl_p * tf.math.log(tcl_p + 1e-10), axis=1, keepdims=True) +
                                  tf.reduce_sum(input_tensor=price_p * tf.math.log(price_p + 1e-10), axis=1, keepdims=True) +
                                  tf.reduce_sum(input_tensor=deficiency_p * tf.math.log(deficiency_p + 1e-10), axis=1, keepdims=True)+
                                  tf.reduce_sum(input_tensor=excess_p * tf.math.log(excess_p + 1e-10), axis=1, keepdims=True))  # maximize entropy (regularization)

        loss_total = tf.reduce_mean(input_tensor=loss_policy + loss_value + entropy)

        optimizer = tf.compat.v1.train.RMSPropOptimizer(LEARNING_RATE)
        minimize = optimizer.minimize(loss_total)

        return s_t, a_t, r_t, minimize, loss_total

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

        s_t, a_t, r_t, minimize, loss = self.graph
        print("Training...")
        LOSS_LIST=[]
        for _ in range(TRAINING_ITERATIONS):
            iter_loss=self.session.run([minimize,loss], feed_dict={s_t: s, a_t: a, r_t: r})[1]
            # LOSS_LIST.append(iter_loss)
        # pyplot.plot(LOSS_LIST)
        # pyplot.show()
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
        global frames;
        frames = frames + 1

        if random.random() < eps:
            p = [np.random.dirichlet(np.ones(NUM_ACTIONS_TCLs),size=1)[0],
                 np.random.dirichlet(np.ones(NUM_ACTIONS_PRICES),size=1)[0],
                 np.random.dirichlet(np.ones(NUM_ACTIONS_DEF),size=1)[0],
                 np.random.dirichlet(np.ones(NUM_ACTIONS_EXCESS),size=1)[0]]
            self.random_action=True
        else:
            s = np.array([s])
            p = brain.predict_p(s)
            self.random_action=False
        a = np.array([np.random.choice(NUM_ACTIONS_TCLs, p=p[0]),np.random.choice(NUM_ACTIONS_PRICES,p=p[1]),np.random.choice(NUM_ACTIONS_DEF,p=p[2]), np.random.choice(NUM_ACTIONS_EXCESS,p=p[3])])
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
        self.env = TCLEnv()
        self.agent = Agent(eps_start, eps_end, eps_steps)


    def runEpisode(self):
        s = self.env.reset()
        R = 0
        while True:
            time.sleep(THREAD_DELAY)  # yield
            if self.render: self.env.render(s)
            a, p = self.agent.act(s)
            s_, r, done, info = self.env.step(a)

            # if done:  # terminal state
            #     s_ = None

            self.agent.train(s, np.concatenate(p), r, s_)

            s = s_
            R += r
            if done:
                if self.render: self.env.render(s)
                break
        # if not self.agent.random_action:
        REWARDS.append(R)
        print(" R:", R)



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
env_test = Environment(render=True, eps_start=0., eps_end=0.)
NUM_STATE = env_test.env.observation_space.shape[0]
NUM_ACTIONS = env_test.env.action_space.shape[0]
NUM_ACTIONS_TCLs = 4
NUM_ACTIONS_PRICES = 5
NUM_ACTIONS_DEF = 2
NUM_ACTIONS_EXCESS = 2

NONE_STATE = np.zeros(NUM_STATE)

brain = Brain()  # brain is global in A3C

envs = [Environment() for i in range(THREADS)]
opts = [Optimizer() for i in range(OPTIMIZERS)]

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
AVGRWRD=[np.average(REWARDS[i:i+10]) for i in range(0,len(REWARDS),10)]
print("Training finished")
pyplot.close("all")
pyplot.plot(AVGRWRD)
pyplot.show()
env_test.env.seed(1)
env_test.run()

# pyplot.plot(REWARDS)
# pyplot.show()

