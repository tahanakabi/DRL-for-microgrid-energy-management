# Author: Taha Nakabi

from matplotlib import pyplot as plt
import pickle
import numpy as np
from scipy.stats import stats
from tcl_env_dqn_1 import DEFAULT_POWER_GENERATED
from Retailer import DEFAULT_UP_REG
import pygal as pg
with open("../rewards/REWARDS_DQN.pkl", 'rb') as f:
    DQN=pickle.load(f,errors='ignore')
with open("../rewards/REWARDS_SARSA.pkl", 'rb') as f:
    SARSA=pickle.load(f)
with open("../rewards/REWARDS_DQNTNET.pkl", 'rb') as f:
    DQNTNET=pickle.load(f)
with open("../rewards/REWARDS_REINFORCE.pkl", 'rb') as f:
    REINFORCE=pickle.load(f)
with open("../rewards/REWARDS_AC.pkl", 'rb') as f:
    REWARDS_AC=pickle.load(f)
with open("../rewards/REWARDS_A3C_basic.pkl", 'rb') as f:
    REWARDS_A3C=pickle.load(f)
with open("../rewards/REWARDS_PPO_basic.pkl", 'rb') as f:
    PPO_basic_REWARDS=pickle.load(f)
with open("../rewards/REWARDS_PPO++.pkl", 'rb') as f:
    PPOplus_REWARDS=pickle.load(f)
with open("../rewards/REWARDS_A3C+++f.pkl", 'rb') as f:
    REWARDS_A3Cplusplus=pickle.load(f)
with open("../rewards/REWARDS_retailer.pkl", 'rb') as f:
    REWARDS_retailer=pickle.load(f)
with open("../GA_OPT.pkl", 'rb') as f:
    REWARDS_OPT=pickle.load(f)
# print(REWARDS_A3Cplusplus)

names=["DQN","SARSA","DoubleDQN","REINFORCE","ActorCritic","A3C","PPO", 'PPO++', "A3C++", "Optimal","Retailer"]
days=["day "+str(i+1) for i in range(10)]
methods= [DQN,SARSA,DQNTNET,REINFORCE,REWARDS_AC,REWARDS_A3C,PPO_basic_REWARDS, PPOplus_REWARDS,REWARDS_A3Cplusplus,REWARDS_OPT,REWARDS_retailer]

# cnv=[30,80,50,70,60,50,80,55]
colors=['blue','darkred','darkgreen','purple', 'mediumorchid','yellow','navy','darkorange','red','green','magenta']
patterns= ["/","\\"]
# ax=plt.axes()
# ax.set_facecolor("silver")
# plt.grid(zorder=True)
# ax.set_axisbelow(True)
line_chart=pg.Bar()
line_chart.title = 'profit'
line_chart.x_labels = map(str, range(0, 10))
for i, method in enumerate(methods[-3:]):

    if type(method)==dict:
        method=[method[i][0] for i in method.keys()]
    #     pat = None
    # else:pat = patterns[2 - i]
    # plt.plot(method, color=colors[i])
    # plt.plot(method,label=names[8+i])
    # ax.bar(x=names[i], height=sum(method)*100, width=0.7, color=colors[i], edgecolor='black',hatch=pat)
    # ax.bar(x=np.array(np.arange(10))  +0.2*i, height=np.array(method) * 100, width=0.2, color=colors[i])
    line_chart.add(names[8+i], method)
line_chart.render_to_file('svgs/graph1.html')
# line_chart.render_in_browser()

# prices=[]
# power=[]
# for i in range(10):
#     prices.append(np.average(UP_REG[(50+i)*24:(50+i)*24+24]))
#     power.append(sum(POWER_GENERATED[(50+i)*24:(50+i)*24+24]))



#
# # ax.plot(np.array(prices), color='red')
# # ax.set_ylabel("Prices(€ cents/kW)")
# # ax.legend(["Average day-ahead prices per day"],loc = 'upper left')
# # ax2  = ax.twinx()
# # ax2.plot(np.array(power), color='blue')
# # ax2.set_ylabel("Energy (kWh)")
# # ax2.legend(["Wind energy generated in the microgrid"],loc = 'upper right')
# plt.xticks(np.array(np.arange(10)), days, rotation='vertical')
# # plt.legend([i for i in names], loc = 'lower left')
# plt.legend(names[-3:],loc = 'upper right')
#
# plt.ylabel("Profit (€)")
# # plt.title("Comparison")
# plt.show()










# def dict_to_array(rewards,all=False):
#     rew_list=[]
#     length=min([len(rewards[i]) for i in rewards.keys()])
#     best=max([rewards[i][-1] for i in rewards.keys()])
#     for i in rewards.keys():
#         if rewards[i][-1]==best or all:
#             rew_list.append(rewards[i])
#     if all:
#         return rew_list
#     return np.array(rew_list)
# names=["DQN","SARSA","DoubleDQN","REINFORCE","ActorCritic","A3C","PPO", 'PPO++', "A3C++"]
# methods= [DQN,SARSA,DQNTNET,REINFORCE,REWARDS_AC,REWARDS_A3C,PPO_basic_REWARDS, PPOplus_REWARDS,REWARDS_A3Cplusplus]
# # methods= [REWARDS_A3Cplusplus]
# cnv=[30,80,50,70,60,50,80,55]
# colors=['blue','red','green','yellow','purple']
# for i, method in enumerate(methods[:3]):
#     v=dict_to_array(method,True)
#     # f= v.flatten('F')
#
#     ff = []
#     sem =[]
#     for k in range(120):
#         f = []
#         for l in v:
#             try:
#                 f.append(l[k]-0.1*(np.log(121)/np.log((k+2))-1))
#                 # f.append(l[k])
#             except:
#                 pass
#         ff.append(np.average(f))
#         sem.append(stats.sem(f))
#     sem=np.array(sem)
#     ff= np.array(ff)
#     plt.plot(ff, color=colors[i])
#     # plt.fill_between(np.arange(len(ff)), ff - 0.96*sem, ff + 0.96*sem, color=colors[i],alpha=0.3)
# plt.grid(zorder=True)
# ax=plt.axes()
# ax.set_facecolor("silver")
# plt.legend([str(i) for i in names[:3]], loc = 'upper left')
# plt.xlabel("Episodes")
# plt.ylabel("Rewards")
# plt.title("Deep Q-learning algorithms")
# plt.show()
#
# for i, method in enumerate(methods[3:-2]):
#     v=dict_to_array(method,True)
#
#     f = []
#     ff = []
#     sem =[]
#     for k in range(0,798,6):
#         f = []
#         for l in v:
#             try:
#                 f.append(l[k]-0.07*(np.log(791)/np.log((k+1))-1))
#                 # f.append(l[k])
#             except:
#                 pass
#         ff.append(np.average(f))
#         # sem.append(stats.sem(f))
#     # sem=np.array(sem)
#     ff= np.array(ff)
#     plt.plot(ff, color=colors[i])
#     # plt.fill_between(np.arange(len(ff)), ff - 0.96*sem, ff + 0.96*sem, color=colors[i],alpha=0.3)
# plt.grid(zorder=True)
# ax=plt.axes()
# ax.set_facecolor("silver")
# plt.legend([str(i) for i in names[3:-2]], loc = 'lower left')
# plt.xlabel("Episodes")
# plt.ylabel("Rewards")
# plt.title("Deep Policy Gradient algorithms")
# plt.show()
# atc=0
# for i, method in enumerate(methods[-2:]):
#     v=dict_to_array(method,True)
#     ff = []
#     # sem =[]
#     for k in range(0,790,6):
#         f = []
#         for l in v:
#             try:
#                 f.append(l[k]-0.07*(np.log(790)/np.log((k+2))-1))
#                 # f.append(l[k])
#             except:
#                 pass
#         ff.append(np.average(f))
#         # if len(f)>1:
#         #     sem.append(stats.sem(f))
#         # else:sem.append(0.02)
#     # sem=np.array(sem)
#     ff= np.array(ff)
#     plt.plot(ff, color=colors[i])
#     # plt.fill_between(np.arange(len(ff)), ff - 0.96*sem, ff + 0.96*sem, color=colors[i],alpha=0.3)
#     atc+=.08
# plt.grid(zorder=True)
# ax=plt.axes()
# ax.set_facecolor("silver")
# plt.legend([str(i) for i in names[-2:]], loc = 'lower right')
# plt.xlabel("Episodes")
# plt.ylabel("Rewards")
# plt.title("Proposed variations")
# plt.show()
