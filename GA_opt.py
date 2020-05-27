import concurrent.futures
import random
import numpy as np
from tcl_env_dqn_1 import *
POP_SHAPE=[100,24,4]
P_C = 0.99 #Probability of crossover
P_M = 0.5 #Probability of mutation

MAX_ITER=200
EARLY_STOP=50
V=[4,5,2,2]


def objective_function(element,render=False):
    # assert (element.shape == POP_SHAPE[1:])
    # print(element.shape)
    enviro = MicroGridEnv(day0=59,dayn=60)
    state = enviro.reset(day=59)
    R=0
    for action in element:
        if render: enviro.render()
        s_, r, done, _ = enviro.step(list(action))
        R += r
    print(R)
    return R


def evaluation(pop):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results=[]
        for p in pop:
            results.append(executor.submit(objective_function, p))
    fitness=[r.result()for r in results]
    return fitness



def initialization():
    """
    Initalizing the population which shape is self.pop_shape(0-1 matrix).
    """

    pop = np.array([np.random.randint(low=0, high=i, size=POP_SHAPE[0:2]) for i in V])
    pop=pop.transpose((1,2,0))

    fitness = evaluation(pop)
    return pop,fitness

def crossover(ind_0, ind_1):
    """
    Single point crossover.
    Args:
        ind_0: individual_0
        ind_1: individual_1
    Ret:
        new_0, new_1: the individuals generated after crossover.
    """
    assert(ind_0.shape == ind_1.shape)

    pointx = np.random.randint(ind_0.shape[0])
    pointy = np.random.randint(ind_0.shape[1])

    if random.random()<0.5:
        new0,new1= horizontal_crossover(ind_0,ind_1,pointx,pointy)
    else:
        new0, new1 = vertical_crossover(ind_0, ind_1, pointx, pointy)

    # assert(new_0.shape == ind_0.shape)
    return new0, new1


def horizontal_crossover(ind_0,ind_1,pointx,pointy):
    new_0 = np.concatenate((ind_0[:pointx,:],np.reshape(np.hstack((ind_0[pointx,:pointy],ind_1[pointx,pointy:])),[1,4]), ind_1[pointx+1:, :]))
    new_1 = np.concatenate((ind_1[:pointx,:],np.reshape(np.hstack((ind_1[pointx,:pointy],ind_0[pointx,pointy:])),[1,4]), ind_0[pointx+1:, :]))
    return new_0,new_1

def vertical_crossover(ind_0,ind_1,pointx,pointy):
    new_0 = np.concatenate((ind_0[:, :pointy],
                            np.reshape(np.hstack((ind_0[:pointx, pointy], ind_1[pointx:, pointy])), [24, 1]),
                            ind_1[:, pointy + 1:]), axis=1)
    new_1 = np.concatenate((ind_1[:, :pointy],
                            np.reshape(np.hstack((ind_1[:pointx, pointy], ind_0[pointx:, pointy])), [24, 1]),
                            ind_0[:, pointy + 1:]), axis=1)
    return new_0, new_1

def mutation(indi):
    """
    Simple mutation.
    Arg:
        indi: individual to mutation.
    """
    pointx = np.random.randint(indi.shape[0])
    for pointy,v in enumerate(V):
        indi[pointx,pointy] = np.random.randint(low=0, high=v)
    return indi

def rws(size, fitness):
    """
    Roulette Wheel Selection.
    Args:
        size: the size of individuals you want to select according to their fitness.
        fitness: the fitness of population you want to apply rws to.
    """
    fitness_ = (fitness-min(fitness))/(max(fitness)-min(fitness))
    probability=fitness_/fitness_.sum()
    idx = np.random.choice(np.arange(len(fitness_)), size=size, replace=True, p=probability)
    return idx

def run():
    """
    Run the genetic algorithm.
    Ret:
        global_best_ind: The best indiviudal during the evolutionary process.
        global_best_fitness: The fitness of the global_best_ind.
    """
    pop, fitness = initialization()
    print(fitness)
    best_index = np.argmax(fitness)
    global_best_fitness = fitness[best_index]
    global_best_indiv = pop[best_index, :]
    eva_times = POP_SHAPE[0]
    count = 0

    for it in range(MAX_ITER):
        next_generation = []
        for n in range(int(POP_SHAPE[0]/2)):
            i, j = rws(2, fitness) # choosing 2 individuals with rws.
            indi_0, indi_1 = pop[i, :].copy(), pop[j, :].copy()
            if np.random.rand() < P_C:
                indi_0, indi_1 = crossover(indi_0, indi_1)

            if np.random.rand() < P_M:
                indi_0 = mutation(indi_0)
            if np.random.rand() < P_M:
                indi_1 = mutation(indi_1)

            next_generation.append(indi_0)
            next_generation.append(indi_1)

        pop = np.array(next_generation)
        fitness = evaluation(pop)
        eva_times += POP_SHAPE[0]

        if np.max(fitness) > global_best_fitness:
            best_index = np.argmax(fitness)
            global_best_fitness = fitness[best_index]
            global_best_indiv = pop[best_index, :]
            count = 0
        else:
            count +=1

        print('Generation {}:'.format(it))
        print('Global best fitness:', global_best_fitness)

        if EARLY_STOP != None and count > EARLY_STOP:
            print('Did not improved within {} rounds. Break.'.format(EARLY_STOP))
            break

    print('\n Solution: {} \n Fitness: {} \n Evaluation times: {}'.format(global_best_indiv, global_best_fitness, eva_times))
    return global_best_indiv, global_best_fitness


if __name__=="__main__":

    # fitness = np.zeros(POP_SHAPE[0])
    # pop = np.zeros(POP_SHAPE)
    # solution , R= run()
    solution=np.array([[0,2,0,1],[0,4,0,1]
            ,[2,2,1,1]
            ,[0,2,0,1]
            ,[0,2,1,0]
            ,[0,4,0,0]
            ,[1,3,1,0]
            ,[1,4,0,1]
            ,[3,3,1,0]
            ,[0,2,0,0]
            ,[1,4,1,1]
            ,[1,4,0,1]
            ,[3,2,1,0]
            ,[2,4,1,1]
            ,[0,2,0,1]
            ,[0,2,0,0]
            ,[0,3,1,1]
            ,[3,4,1,1]
            ,[3,4,1,0]
            ,[1,2,1,1]
            ,[0,3,0,0]
            ,[0,2,1,1]
            ,[2,3,1,0]
            ,[3,4,1,1]])
    # solution50=np.array([[1, 1, 0, 1]
    #     , [1, 0, 0, 1]
    #     , [3, 0, 1, 1]
    #     , [2, 1, 0, 1]
    #     , [1, 0, 0, 1]
    #     , [2, 1, 1, 1]
    #     , [1, 2, 0, 1]
    #     , [3, 2, 0, 0]
    #     , [0, 3, 0, 0]
    #     , [1, 3, 0, 1]
    #     , [3, 3, 1, 0]
    #     , [3, 2, 1, 0]
    #     , [0, 2, 0, 1]
    #     , [2, 2, 0, 1]
    #     , [2, 3, 1, 0]
    #     , [3, 3, 1, 1]
    #     , [0, 4, 1, 1]
    #     , [3, 3, 1, 0]
    #     , [2, 3, 0, 0]
    #     , [1, 3, 0, 0]
    #     , [3, 3, 0, 1]
    #     , [1, 3, 1, 0]
    #     , [3, 3, 1, 0]
    #     , [2, 4, 0, 0]])
    objective_function(solution,render=True)
