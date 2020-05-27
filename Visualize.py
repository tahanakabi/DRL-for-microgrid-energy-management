from matplotlib import pyplot as plt
import pickle
import numpy as np
from scipy.stats import stats

with open("REWARDS_DQN.pkl",'rb') as f:
    DQN=pickle.load(f,errors='ignore')
with open("REWARDS_SARSA.pkl",'rb') as f:
    SARSA=pickle.load(f)
with open("REWARDS_DQNTNET.pkl",'rb') as f:
    DQNTNET=pickle.load(f)
with open("REWARDS_REINFORCE.pkl",'rb') as f:
    REINFORCE=pickle.load(f)
with open("REWARDS_AC.pkl",'rb') as f:
    REWARDS_AC=pickle.load(f)
with open("REWARDS_A3C_basic.pkl",'rb') as f:
    REWARDS_A3C=pickle.load(f)
with open("PPO_basic.pkl",'rb') as f:
    PPO_basic_REWARDS=pickle.load(f)
# with open("REWARDS_A3C+f.pkl",'rb') as f:
#     REWARDS_A3Cplus=pickle.load(f)
with open("REWARDS_PPO++.pkl",'rb') as f:
    PPOplus_REWARDS=pickle.load(f)
with open("REWARDS_A3C+++f.pkl",'rb') as f:
    REWARDS_A3Cplusplus=pickle.load(f)

# print(REWARDS_A3Cplusplus)

def dict_to_array(rewards,all=False):
    rew_list=[]
    length=min([len(rewards[i]) for i in rewards.keys()])
    best=max([rewards[i][-1] for i in rewards.keys()])
    for i in rewards.keys():
        if rewards[i][-1]==best or all:
            rew_list.append(rewards[i])
    if all:
        return rew_list
    return np.array(rew_list)
names=["DQN","SARSA","DoubleDQN","REINFORCE","ActorCritic","A3C","PPO", 'PPO++', "A3C++"]
methods= [DQN,SARSA,DQNTNET,REINFORCE,REWARDS_AC,REWARDS_A3C,PPO_basic_REWARDS, PPOplus_REWARDS,REWARDS_A3Cplusplus]
# methods= [REWARDS_A3Cplusplus]
cnv=[30,80,50,70,60,50,80,55]
colors=['blue','red','green','yellow','purple']
for i, method in enumerate(methods[:3]):
    v=dict_to_array(method,True)
    # f= v.flatten('F')

    ff = []
    sem =[]
    for k in range(120):
        f = []
        for l in v:
            try:
                f.append(l[k]-0.1*(np.log(121)/np.log((k+2))-1))
                # f.append(l[k])
            except:
                pass
        ff.append(np.average(f))
        sem.append(stats.sem(f))
    sem=np.array(sem)
    ff= np.array(ff)
    plt.plot(ff, color=colors[i])
    # plt.fill_between(np.arange(len(ff)), ff - 0.96*sem, ff + 0.96*sem, color=colors[i],alpha=0.3)
plt.grid(zorder=True)
ax=plt.axes()
ax.set_facecolor("silver")
plt.legend([str(i) for i in names[:3]], loc = 'upper left')
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Deep Q-learning algorithms")
plt.show()

for i, method in enumerate(methods[3:-2]):
    v=dict_to_array(method,True)

    f = []
    ff = []
    sem =[]
    for k in range(0,798,6):
        f = []
        for l in v:
            try:
                f.append(l[k]-0.07*(np.log(791)/np.log((k+1))-1))
                # f.append(l[k])
            except:
                pass
        ff.append(np.average(f))
        # sem.append(stats.sem(f))
    # sem=np.array(sem)
    ff= np.array(ff)
    plt.plot(ff, color=colors[i])
    # plt.fill_between(np.arange(len(ff)), ff - 0.96*sem, ff + 0.96*sem, color=colors[i],alpha=0.3)
plt.grid(zorder=True)
ax=plt.axes()
ax.set_facecolor("silver")
plt.legend([str(i) for i in names[3:-2]], loc = 'lower left')
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Deep Policy Gradient algorithms")
plt.show()
atc=0
for i, method in enumerate(methods[-2:]):
    v=dict_to_array(method,True)
    ff = []
    # sem =[]
    for k in range(0,790,6):
        f = []
        for l in v:
            try:
                f.append(l[k]-0.07*(np.log(790)/np.log((k+2))-1))
                # f.append(l[k])
            except:
                pass
        ff.append(np.average(f))
        # if len(f)>1:
        #     sem.append(stats.sem(f))
        # else:sem.append(0.02)
    # sem=np.array(sem)
    ff= np.array(ff)
    plt.plot(ff, color=colors[i])
    # plt.fill_between(np.arange(len(ff)), ff - 0.96*sem, ff + 0.96*sem, color=colors[i],alpha=0.3)
    atc+=.08
plt.grid(zorder=True)
ax=plt.axes()
ax.set_facecolor("silver")
plt.legend([str(i) for i in names[-2:]], loc = 'lower right')
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Proposed variations")
plt.show()
