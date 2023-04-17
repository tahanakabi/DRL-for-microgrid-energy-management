# A3C++ a modified version of Asynchronous Advantage actor critic algorithm
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

# Args: 'train' for training the model anything else will skip the training and try to use already saved models

import tensorflow as tf
import numpy as np
import gym, time, random, threading
from keras.callbacks import TensorBoard
from keras.models import *
from keras.layers import *
from keras import backend as K

from tcl_env_dqn_1 import *
print("after import")
import os



# This is where the models are saved and retrieved from
MODELS_DIRECTORY = 'success01'
# For tensor board
NAME= "A3C++logs/A3C++{}".format(int(time.time()))
# -- constants
# Threading parameters
RUN_TIME = 5000
THREADS = 16
OPTIMIZERS = 2
THREAD_DELAY = 0.000001
# Reinforcement learning parameters
N_STEP_RETURN = 15
GAMMA = 1.0
GAMMA_N = GAMMA ** N_STEP_RETURN
# Epsilon greedy strategy parameters
EPS_START = .5
EPS_STOP = .001
EPS_DECAY = 5e-6
# Memory replay parameters
MIN_BATCH = 200
TR_FREQ = 100
# Advantage actor-critic parameters
LOSS_V = 0.4  # v loss coefficient
LOSS_ENTROPY = 1.0  # entropy coefficient
# Initializing max rewards for models' saving purposes
max_reward = -100.0
# Training iterations and learning rate
TRAINING_ITERATIONS = 1
LEARNING_RATE = 1e-3
# ---------
# The brain class will handle building the neural network, sampling experiences for training and preparing and running the training process.
# ---------
class Brain:
    # Memory
    train_queue = [[], [], [], [], []]  # s, a, r, s', s' terminal mask
    train_queue_copy = [[], [], [], [], []]  # s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()

    def __init__(self, **kwargs):
        self.env = kwargs.get("environment")
        self.learning_rate = kwargs.get('learning_rate', LEARNING_RATE)
        self.tr_freq = kwargs.get('training_frequency', TR_FREQ)
        self.min_batch = kwargs.get('min_batch', MIN_BATCH)
        self.gamman = kwargs.get('gamma_n', GAMMA_N)
        self.models_directory = kwargs.get('models_directory', MODELS_DIRECTORY)
        self.num_state = self.env.env.observation_space.shape[0]
        self.num_tcl =self.env.env.num_tcls
        self.num_actions= self.env.env.action_space.n
        self.none_state=np.zeros(self.num_state)
        tf.compat.v1.disable_eager_execution()
        # self.session = tf.compat.v1.Session()
        # K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.model = self._build_model(num_state=self.num_state, num_tcls=self.num_tcl)
        self.graph = self._build_graph(self.model)
        # self.session.run(tf.compat.v1.global_variables_initializer())
        # self.default_graph = tf.compat.v1.get_default_graph()
        # We keep track of the best rewards achieved so far for each day
        self.max_reward = max_reward
        self.rewards = {}
        for i in range(self.env.env.day0, self.env.env.dayn):
            self.rewards[i] = self.max_reward

        # self.default_graph.finalize()  # avoid modifications

    def _build_model(self, num_state, num_tcls):

        l_input = Input(batch_shape=(None,num_state))
        print('input shape')
        print(format(l_input.shape.as_list()))
        # The TCLs states are fed individually to the neural network but they are simply being averaged
        l_input1 = Lambda(lambda x: x[:, 0:num_tcls])(l_input)
        l_input2 = Lambda(lambda x: x[:, num_tcls:])(l_input)
        print(self.env.env.num_tcls)
        l_input1 = Reshape((num_tcls, 1))(l_input1)
        l_Pool = AveragePooling1D(pool_size=num_tcls)(l_input1)
        l_Pool = Reshape([1])(l_Pool)
        l_dense = Concatenate()([l_Pool, l_input2])
        l_dense = Dense(100, activation='relu')(l_dense)
        l_dense = Dropout(0.3)(l_dense)
        out = Dense(self.num_actions, activation='softmax')(l_dense)
        out_value = Dense(1, activation='linear')(l_dense)
        model = Model(inputs=l_input, outputs=[out, out_value])
        model._make_predict_function()  # have to initialize before threading
        return model

    def _build_graph(self, model):
        s_t = tf.compat.v1.placeholder(tf.float32, shape=(None, self.num_state))
        a_t = tf.compat.v1.placeholder(tf.float32, shape=(None, self.num_actions))
        r_t = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))  # not immediate, but discounted n step reward
        p, v = model(s_t)
        log_prob = tf.math.log(tf.reduce_sum(input_tensor=p * a_t, axis=1, keepdims=True) + 1e-10)
        advantage = r_t - v
        loss_policy =  -log_prob * tf.stop_gradient(advantage)  # maximize policy
        loss_value = LOSS_V * tf.square(advantage)  # minimize value error
        entropy = LOSS_ENTROPY * (tf.reduce_sum(input_tensor=p * tf.math.log(p + 1e-10), axis=1, keepdims=True))
        loss_total = tf.reduce_mean(input_tensor=loss_policy + loss_value + entropy)
        optimizer = tf.compat.v1.train.RMSPropOptimizer(self.learning_rate)
        minimize = optimizer.minimize(loss_total)
        return s_t, a_t, r_t, minimize, loss_total

    def optimize(self):
        # self.train_queue_copy serves as a counter of the number of observations we make between training sessions
        if len(self.train_queue_copy[0])<self.tr_freq or len(self.train_queue_copy[0])<self.min_batch :
            time.sleep(0)  # yield
            return

        with self.lock_queue:
            if len(self.train_queue_copy[0])<self.tr_freq:  # more thread could have passed without lock
                return  # we can't yield inside lock
            # We take a fraction from the memory and throw away the rest, the following experiences are added on top of the sampled experiences.
            # This sampling process makes the current memory include old and new experiences. After many sampling iterations the very old experiences will slowly fade and the newest will remain.
            self.train_queue = random.sample(np.array(self.train_queue).T.tolist(), self.min_batch)
            self.train_queue = np.array(self.train_queue).T.tolist()
            s, a, r, s_, s_mask = self.train_queue_copy
            self.train_queue_copy = [[], [], [], [], []]

            s = np.vstack(s)
            a = np.vstack(a)
            r = np.vstack(r)
            s_ = np.vstack(s_)
            s_mask = np.vstack(s_mask)

        if len(s) > 5 * self.min_batch: print("Optimizer alert! Minimizing batch of %d" % len(s))

        v = self.predict_v(s_)
        r = r + self.gamman * v * s_mask  # set v to 0 where s_ is terminal state

        s_t, a_t, r_t, minimize, loss = self.graph
        print("Training...")
        # for _ in range(TRAINING_ITERATIONS):
        minimize(s,a,r)
        # self.session.run([minimize,loss], feed_dict={s_t: s, a_t: a, r_t: r})
        print("Done...")


    # pushing experiences into the memory
    def train_push(self, s, a, r, s_):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            self.train_queue_copy[0].append(s)
            self.train_queue_copy[1].append(a)
            self.train_queue_copy[2].append(r)

            if s_ is None:
                self.train_queue[3].append(self.none_state)
                self.train_queue[4].append(0.)

                self.train_queue_copy[3].append(self.none_state)
                self.train_queue_copy[4].append(0.)
            else:
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)

                self.train_queue_copy[3].append(s_)
                self.train_queue_copy[4].append(1.)

    def predict(self, s):
        # with self.default_graph.as_default():
        p, v = self.model.predict(s)
        return p, v

    def predict_p(self, s):
        # with self.default_graph.as_default():
        p, v = self.model.predict(s)
        return p

    def predict_p_vote(self, s):
        # Boost learning. Several versions of the successfull models are voting for the best action
        votes=[]
        # print('retreiving models from {}'.format(self.models_directory))
        for filename in os.listdir(self.models_directory):
            if filename.endswith(".h5"):
                # print(filename)
                # with self.default_graph.as_default():
                try:
                    # print('trying to load weights')
                    self.model.load_weights(self.models_directory+"/"+filename)
                    # print('weights loaded')
                    p = self.model.predict(s)[0][0]
                    # print('probability predicted')
                    # votes.append(p)
                    votes.append(ACTIONS[np.argmax(p)])
                except :
                    print(filename+"didn't vote!")
                    pass
        boosted_p = np.average(np.array(votes),axis=0)
        return  np.rint(boosted_p).astype(int)
        # return ACTIONS[np.argmax(boosted_p)]

    def predict_v(self, s):
        # with self.default_graph.as_default():
        p, v = self.model.predict(s)
        return v

# ---------
# The agent handles the interactions with the environment and the selection of actions, stocking and retreiving experiences from the memory.
# ---------
frames = 0

class Agent:
    def __init__(self, eps_start, eps_end, eps_decay, num_actions):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.memory = []  # used for n_step return
        self.R = 0.
        self.num_actions = num_actions

    def getEpsilon(self):
        return max(self.eps_start -  frames * self.eps_decay,self.eps_end)  # linearly interpolate

    def act(self, s,render=False, br=None):
        global frames, brain
        if br != None:
            brain = br
        eps = self.getEpsilon()
        frames = frames + 1
        # Epsilon-greedy strategy:
        if random.random() < eps:
            p = np.random.dirichlet(np.ones(self.num_actions), size=1)
        else:
            s = np.array([s])
            if render:
                print('starting the vote')
                a = brain.predict_p_vote(s)
                p= np.random.dirichlet(np.ones(self.num_actions), size=1)
                print(a)
                return list(a),p
            p = brain.predict_p(s)
        # In the original version, the action selection follows a stochasic policy as follows:
        # a = np.random.choice(NUM_ACTIONS, p=p.reshape(NUM_ACTIONS,))
        # We follow a deterministic policy as follow:
        a = np.argmax(p.reshape(self.num_actions,))
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
# The environment here is defined as a thread so that we can run the algorithm as a multi-thread process
# ---------
class Environment(threading.Thread):
    stop_signal = False

    def __init__(self, render=False, eps_start=EPS_START, eps_end=EPS_STOP, eps_decay=EPS_DECAY, **kwargs):
        threading.Thread.__init__(self)

        self.render = render
        self.env = MicroGridEnv(**kwargs)
        self.agent = Agent(eps_start, eps_end, eps_decay,num_actions=self.env.action_space.n)
        self.brain = None


    def runEpisode(self,day=None, pplt=True, web = False):
        # print('resetting the environment')
        if web==False:
            s = self.env.reset_all(day=day)
        else:
            s = self.env.reset(day=day)
        R = 0
        while True:
            time.sleep(THREAD_DELAY)  # yield
            # print('Acting')
            a, p = self.agent.act(s,self.render, self.brain)
            # print('stepping')
            s_, r, done, _ = self.env.step(a)
            R += r
            # print('rendering')
            if self.render:
                self.env.render(R)

            if done:  # terminal state
                s_ = None

            if not self.render:
                aa = np.zeros(shape=(NUM_ACTIONS,))
                aa[a] = 1
                self.agent.train(s, aa, r, s_)
            s = s_

            if done:
                break
        print("episode has been ran")
        print(R)
        if web==False:
            REWARDS[self.env.day].append(R)

        if self.render:
            return R
        if R > brain.rewards[self.env.day] and  self.agent.getEpsilon()<0.2:
            print('new max found: '+str(R))
            print("-------------------------------------------------------------------------------------------------")
            try:
                # Uncomment the following line for tensorboard
                writer = tf.compat.v1.summary.FileWriter(NAME, brain.session.graph)
                brain.model.save(MODELS_DIRECTORY+"/A3C++" + str(self.env.day) + ".h5")
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




if __name__ =="__main__":
    import sys
    TRAIN=False
    if str(sys.argv[1]) == 'train':
        TRAIN = True

    DAY0 = 0
    DAYN = 10

    REWARDS = {}
    for i in range(DAY0,DAYN):
        REWARDS[i]=[]

    env_test = Environment(render=True, eps_start=0., eps_end=0., day0=DAY0, dayn=DAYN, iterations=100)
    NUM_STATE = env_test.env.observation_space.shape[0]
    NUM_ACTIONS = env_test.env.action_space.n
    NONE_STATE = np.zeros(NUM_STATE)

    brain = Brain(environment=env_test)  # brain is global in A3C

    if TRAIN:

        envs = [Environment(day0=DAY0, dayn=DAYN) for i in range(THREADS)]
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
        brain.model.save("success00/A3C++"  + ".h5")
        print("Training finished")
        print('training_time:', time.time()-t0)
        # Save the rewards' list for each day
        import pickle
        with open("REWARDS_A3C++train.pkl", 'wb') as f:
            pickle.dump(REWARDS, f, pickle.HIGHEST_PROTOCOL)


    try:
        for day in range(DAY0,DAYN):
            env_test.runEpisode(day)
        print("average reward: ",np.average([list(REWARDS[i])[-1] for i in range(DAY0,DAYN)]))
        import pickle
        # with open("REWARDS_A3C++test.pkl", 'wb') as f:
        #     pickle.dump(REWARDS, f, pickle.HIGHEST_PROTOCOL)
    except NameError:
        print(NameError)



