import numpy as np

POP_SHAPE=[100,100]
P_C = 0.8 #Probability of crossover
P_M = 0.2 #Probability of mutation

MAX_ITER=1000
EARLY_STOP=50

def objective_function(element):
    assert (element.shape[0] == POP_SHAPE[1])
    return sum(element[:50])-sum(element[50:])+50

def evaluation(pop):
    return np.array([objective_function(i) for i in pop])


def initialization():
    """
    Initalizing the population which shape is self.pop_shape(0-1 matrix).
    """
    pop = np.random.randint(low=0, high=2, size=POP_SHAPE)
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
    assert(len(ind_0) == len(ind_1))

    point = np.random.randint(len(ind_0))
    new_0 = np.hstack((ind_0[:point], ind_1[point:]))
    new_1 = np.hstack((ind_1[:point], ind_0[point:]))

    assert(len(new_0) == len(ind_0))
    return new_0, new_1

def mutation(indi):
    """
    Simple mutation.
    Arg:
        indi: individual to mutation.
    """
    point = np.random.randint(len(indi))
    indi[point] = 1 - indi[point]
    return indi

def rws(size, fitness):
    """
    Roulette Wheel Selection.
    Args:
        size: the size of individuals you want to select according to their fitness.
        fitness: the fitness of population you want to apply rws to.
    """
    fitness_ = 1.0 / fitness
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
    best_index = np.argmin(fitness)
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
                indi_1 = mutation(indi_1)

            next_generation.append(indi_0)
            next_generation.append(indi_1)

        pop = np.array(next_generation)
        fitness = evaluation(pop)
        eva_times += POP_SHAPE[0]

        if np.min(fitness) < global_best_fitness:
            best_index = np.argmin(fitness)
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

    fitness = np.zeros(POP_SHAPE[0])
    pop = np.zeros(POP_SHAPE)
    run()