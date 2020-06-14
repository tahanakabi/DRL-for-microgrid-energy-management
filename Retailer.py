# Author: Taha Nakabi

from tcl_env_dqn_1 import *
from matplotlib import pyplot
DAY0=50
DAYN=60
# pyplot.plot(UP_REG[DAY0*24:DAYN*24+24])
pyplot.show()
UP_REG = np.genfromtxt("day_ahead_prices2018.csv", delimiter=';', skip_header=1) / 10
# pyplot.plot(UP_REG[DAY0*24:DAYN*24+24])
pyplot.show()
POWER_GENERATED = np.zeros(UP_REG.shape)
TCL_SALE_PRICE = DEFAULT_MARKET_PRICE



ACTION=[0,2,0,0]

def daily_margin(day, render=False):
    enviro = MicroGridEnv(day0=day,dayn=day+1)
    state = enviro.reset(day=day)
    R=0
    for _ in range(24):
        if render: enviro.render()
        s_, r, done, _ = enviro.step(list(ACTION))
        R += r
    print(R)
    return R

REWARDS=[]
for day in range(DAY0,DAYN):
    R=daily_margin(day)
    REWARDS.append(R)
import pickle
with open("REWARDS_retailer.pkl", 'wb') as f:
    pickle.dump(REWARDS, f, pickle.HIGHEST_PROTOCOL)
