# DQN a modified version of DQN algorithm
# To solve the problem of migrogrid's energy management
# -----------------------------------

# The DQN implementation is available at:
# https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/
# by: Jaromir Janisch, 2017
# Adapted to solve the problem of microgrid energy management

# Author: Taha Nakabi

import numpy

# -------------------- BRAIN ---------------------------
from keras.optimizers import *
from keras.models import *
from keras.layers import *

DAY0 = 50
DAYN = 60
REWARDS = {}
for i in range(DAY0,DAYN,1):
    REWARDS[i]=[]

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.model = self._createModel()

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

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)


# -------------------- AGENT ---------------------------
MEMORY_CAPACITY = 500
BATCH_SIZE = 200

GAMMA = 1.0


# MAX_EPSILON = 0.4
# MIN_EPSILON = 0.004
# LAMBDA =5e-5  # speed of decay


class Agent:
    steps = 0
    # epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)

    def act(self, s, deter):
        if deter == True:
            return numpy.argmax(self.brain.predictOne(s))
        # if random.random() < self.epsilon:
        return random.randint(0, self.actionCnt - 1)
        # return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

        # # slowly decrease Epsilon based on our eperience
        # self.steps += 1
        # self.epsilon = max(MAX_EPSILON -LAMBDA * self.steps, MIN_EPSILON)
        # print(self.epsilon)

    def replay(self):
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

        self.brain.train(x, y)


# -------------------- ENVIRONMENT ---------------------
from scripts.gymEnvironment.tcl_env_dqn_1 import *

class Environment:
    def __init__(self, render = False):
        self.env = MicroGridEnv()
        self.render=render


    def run(self, agent, day=None):
        s = self.env.reset(day0=DAY0, dayn=DAYN, day= day)
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
        print("Day ", self.env.day)
        print("R= ", R)


# -------------------- MAIN ----------------------------
if __name__=="__main__":
    # PROBLEM = TCLEnv
    env = Environment()


    stateCnt = env.env.observation_space.shape[0]
    actionCnt = env.env.action_space.n
    agent = Agent(stateCnt, actionCnt)

    import pickle
    import time
    t0=time.time()
    # for _ in range(1000):
    #     env.run(agent)
    # print('training_time:', time.time()-t0)
    # agent.brain.model.save_weights("DQN.h5")
    # with open("REWARDS_DQN.pkl",'wb') as f:
    #     pickle.dump(REWARDS,f,pickle.HIGHEST_PROTOCOL)
    # for rew in REWARDS.values():
    #     print(np.average(list(rew)))
    #     pyplot.plot(list(rew))
    # pyplot.legend(["Day {}".format(i) for i in range(DAY0,DAY0)], loc = 'upper right')
    # pyplot.show()
    agent.brain.model.load_weights("DQN.h5")
    env_test=Environment(render=True)
    for day in range(DAY0,DAYN):
        env_test.run(agent,day=day)
    print(np.average([list(REWARDS[i])[-1] for i in range(DAY0,DAYN)]))
    with open("../../rewards/REWARDS_DQN.pkl", 'wb') as f:
        pickle.dump(REWARDS,f,pickle.HIGHEST_PROTOCOL)
