# Author: Taha Nakabi
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import stats

names=['SARSA', 'A3C', 'A3C+++','PPO', 'PPO+++','baseline1','baseline2']
actual_names=['SARSA', 'A3C', 'A3C++','PPO', 'PPO++','Baseline1','Baseline2']
graphs=['Cost', 'Energy_bought_sold','TOTAL_Consumption']
actual_graphs=['Average energy cost per day', 'Energy exchanged with the grid per hour','Total energy consumption per hour ','Peak shaving performance']
namess=[['SARSA'], ['A3C', 'A3C+'],['PPO++', 'PPO+++'],['baseline1','baseline2']]
actual_namess=[['SARSA'], ['A3C', 'A3C++'],['PPO', 'PPO++'],['Baseline1','Baseline2']]

# for i, nm in enumerate(names):
#     datas=[]
#     for day in range(11):
#         datas.append(np.load(nm+'Cost'+str(day)+'.npy'))
#     datas=np.array(datas)
#
#     iterations=actual_names
#     plt.bar(x=actual_names[i] , height=np.average(datas), width=0.15)
# # plt.grid(zorder=True)
# ax = plt.axes()
# ax.set_facecolor("silver")
# ax.set_ylabel("€")
# ax.yaxis.grid(True)
# # plt.legend(actual_names, loc='lower right')
# plt.title(actual_graphs[0])
# plt.show()
#
#
#
# ax = plt.axes()
# ax.set_facecolor("silver")
# ax.set_ylabel("kW")
# ax.yaxis.grid(True)
# datas=[]
# for i, nm in enumerate(names):
#     algo=[]
#     for day in range(11):
#         algo.append(np.load(nm + 'Energy_bought_sold' + str(day) + '.npy'))
#     algo=np.reshape(np.array(algo),np.array(algo).shape[0]*np.array(algo).shape[1])
#     datas.append(algo)
# datas=np.array(datas)
# bplot=ax.boxplot(x=datas.transpose(), labels=actual_names,vert=True, patch_artist=True,)
#
colors=['royalblue','darkorange','green','red','mediumorchid','saddlebrown','hotpink']
colorss=[['royalblue'],['darkorange','green'],['red','mediumorchid'],['saddlebrown','hotpink']]
#
# for patch, color in zip(bplot['boxes'], colors):
#         patch.set_facecolor(color)
#
#
# # plt.legend(actual_names, loc='upper right')
# plt.title(actual_graphs[1])
# plt.show()
#
#
#
#
#
# markers=['--','-','+--','o--','*--']
#


# fig=plt.figure()
# for k,nms in enumerate(namess):
#     ax1 = fig.add_subplot(2, 2, k + 1)
#     ax1.set_facecolor("silver")
#     ax1.set_ylabel("kW")
#     ax1.set_xlabel("Time (h)")
#     ax1.yaxis.grid(True)
#     for i, nm in enumerate(nms):
#         datas=[]
#         for day in range(11):
#             datas.append(np.load(nm + 'TOTAL_Consumption' + str(day) + '.npy'))
#         datas=np.array(datas)
#         sem=stats.sem(datas)
#         avg =np.average(datas, axis=0)
#         # plt.bar(x=np.arange(len(avg))+0.08*i, height=avg, width=0.08)
#         ax1.plot(avg,label=actual_namess[k][i],color=colorss[k][i])
#         ax1.fill_between(np.arange(len(sem)),  avg- 0.96 * sem, avg + 0.96 * sem, color=colorss[k][i], alpha=0.3)
#     ax1.legend( loc='upper right')
# fig.suptitle(actual_graphs[2])
# plt.show()




#
averages=[]
for i, nm in enumerate(names):
    datas=[]
    avgs=[]
    for day in range(11):
        data = np.load(nm + 'TOTAL_Consumption' + str(day) + '.npy')
        avgs.append(np.average(data))
    averages.append(avgs)
averages = np.array(averages)
averages=np.average(averages,axis=0)

print(averages)




ax = plt.axes()
ax.set_facecolor("silver")
ax.yaxis.grid(True)

for i, nm in enumerate(names):
    datas=[]
    indices=[]
    avgs=[]
    for day in range(11):
        data = np.load(nm + 'TOTAL_Consumption' + str(day) + '.npy')
        ilf = averages[day]/np.max(data)
        ipv = (np.max(data)-np.min(data))/np.max(data)
        # ipv=np.max(data)
        # ip = np.average(data)/np.average(data[6:10])
        indices.append([ilf,ipv])
    indices=np.array(indices)
    datas=np.average(indices,axis=0)
    ax.bar(x=np.arange(2)+0.08*i, height=datas, width=0.05)
ax.set_xticks(np.arange(0.2,2.2,1))
ax.set_xticklabels(['ILF', 'IPV','IP'])
plt.legend(actual_names, loc='lower right')
plt.title(actual_graphs[3] )
plt.show()
# #
# #
# #
