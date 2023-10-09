# Author: Taha Nakabi


from DQN import  Environment
PRICES_ACTIONS=[0,0,0,0,1,2,3,4,4,3,2,1,2,1,2,0,0,0,0,0,0,0,0,0]
import numpy as np

REWARDS = {}
for i in range(11):
    REWARDS[i]=[]


class Agentb1:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
    def act(self, s,deter):

        return [0,PRICES_ACTIONS[int(s[-1]*24)],1,1]
    def observe(self, sample):
        pass
    def replay(self):
        pass
class Environmentb1(Environment):
    def __init__(self, render = False):
        super().__init__(render)


    def run(self, agent, day=None):
        s = self.env.reset(day=day)
        R = 0
        while True:

            if self.render: self.env.render(name='baseline1')

            a = agent.act(s,deter=self.render)

            s_, r, done, info = self.env.step(a)

            if done:  # terminal state
                s_ = None
            s = s_
            R += r
            if done:
                if self.render: self.env.render(name='baseline1')
                break
        REWARDS[self.env.day].append(R)
        print("Total reward:", R)







if __name__=="__main__":
    env_test=Environmentb1(render=True)
    stateCnt = env_test.env.observation_space.shape[0]
    actionCnt = env_test.env.action_space.n
    agent = Agentb1(stateCnt, actionCnt)

    # for day in range(11):
    env_test.run(agent,day=3)
    # print(np.average([REWARDS[i] for i in range(11)]))