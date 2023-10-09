# Author: Taha Nakabi
import os
# names=["DQN","SARSA","DoubleDQN","REINFORCE","ActorCritic","A3C_basic","PPO_basic", 'PPO', "A3C_plusplus"]
# for script in names[:-1]:
#     os.system(script+".py")
#
# os.system(names[-1]+".py test")



# DAY0 = 50
# DAYN = 60
#
# REWARDS = {}
# for i in range(DAY0,DAYN):
#     REWARDS[i]=[]
#
# env_test = Environment(render=True, eps_start=0., eps_end=0.)
# NUM_STATE = env_test.env.observation_space.shape[0]
# NUM_ACTIONS = env_test.env.action_space.n
# NUM_ACTIONS_TCLs = 4
# NUM_ACTIONS_PRICES = 5
# NUM_ACTIONS_DEF = 2
# NUM_ACTIONS_EXCESS = 2
#
# NONE_STATE = np.zeros(NUM_STATE)
# brain = Brain()  # brain is global in A3C