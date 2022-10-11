from utils import Utils, Log, GPUTools
from population import initialize_population
from evaluate import decode, fitnessEvaluate
from evolve import FDE
import copy, os
from datetime import datetime

def create_directory():
    dirs = ['./log', './populations', './scripts', './datasets']
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def fitness_evaluate(population, curr_gen, idx):
    filenames = []
    if idx:
        filename = decode(population[0], curr_gen, idx)
        filenames.append(filename)
    else:
        for i, individual in enumerate(population):
            filename = decode(individual, curr_gen, i)
            filenames.append(filename)
    acc_set, num_parameters = fitnessEvaluate(filenames, curr_gen, is_test=False)

    return acc_set, num_parameters

def evolve1(population, acc_set, num_parameters, params):
    new_pop = []
    acc_set_new = []
    num_parameters_new = []

    offspring = []
    for i, individual in enumerate(population):
        individual_ = copy.deepcopy(individual)
        fde = FDE(individual_, i, population, acc_set, params, Log)
        indi_mut = fde.mutate()
        indi_new = fde.crossover(indi_mut)
        offspring.append(indi_new)

    offspring_new = 0
    acc_set_offspring, num_parameters_offspring = fitness_evaluate(offspring, params['gen_no'], None)
    for i, acc in enumerate(acc_set_offspring):
        if acc >= acc_set[i]:
            new_pop.append(offspring[i])
            acc_set_new.append(acc)
            num_parameters_new.append(num_parameters_offspring[i])
            offspring_new +=1
        else:
            new_pop.append(population[i])
            acc_set_new.append(acc_set[i])
            num_parameters_new.append(num_parameters[i])
    Log.info('EVOLVE[%d-gen]-%d offspring are generated, %d enter into next generation' % (params['gen_no'], len(offspring), offspring_new))
    return new_pop, acc_set_new, num_parameters

def save_record(_str, first_time):
    dt = datetime.now()
    dt.strftime('%Y-%m-%d %H:%M:%S')
    if first_time:
        file_mode = 'w'
    else:
        file_mode = 'a+'
    f = open('./populations/pop_update.txt', file_mode)
    f.write('[%s]-%s\n' % (dt, _str))
    f.flush()
    f.close()

def evolve(population, acc_set, num_parameters, params):
    new_pop = []
    acc_set_new = []
    num_parameters_set_new = []

    offspring_new = 0
    for i, individual in enumerate(population):
        individual_ = copy.deepcopy(individual)
        fde = FDE(individual_, i, population, acc_set, params, Log)
        indi_mut = fde.mutate()
        indi_new = fde.crossover(indi_mut)
        acc_new, num_parameters_new = fitness_evaluate([indi_new], params['gen_no'], i)

        if acc_new >= acc_set[i]:
            new_pop.append(indi_new)
            acc_set_new.append(acc_new[0])
            num_parameters_set_new.append(num_parameters_new[0])
            population[i] = indi_new
            offspring_new += 1
        else:
            new_pop.append(population[i])
            acc_set_new.append(acc_set[i])
            num_parameters_set_new.append(num_parameters[i])

    _str = 'EVOLVE[%d-gen]-%d offspring are generated, %d enter into next generation' % (params['gen_no'], len(new_pop), offspring_new)
    Log.info(_str)
    if params['gen_no'] <= 1:
        save_record(_str, first_time=True)
    else:
        save_record(_str, first_time=False)
    return new_pop, acc_set_new, num_parameters_set_new

def update_best_individual(population, acc_set, num_parameters, gbest):
    if not gbest:
        pbest_individuals = copy.deepcopy(population)
        pbest_accSet = copy.deepcopy(acc_set)
        pbest_params = copy.deepcopy(num_parameters)
        gbest_individual, gbest_acc, gbest_params = getGbest([pbest_individuals, pbest_accSet, pbest_params])
    else:
        gbest_individual, gbest_acc, gbest_params = gbest
        for i,acc in enumerate(acc_set):
            if acc > gbest_acc:
                gbest_individual = copy.deepcopy(population[i])
                gbest_acc = copy.deepcopy(acc)
                gbest_params = copy.deepcopy(num_parameters[i])

    return [gbest_individual, gbest_acc, gbest_params]

def getGbest(pbest):
    pbest_individuals, pbest_accSet, pbest_params = pbest
    gbest_acc = 0
    gbest_params = 1e9
    gbest = None
    for i,indi in enumerate(pbest_individuals):
        if pbest_accSet[i] > gbest_acc:
            gbest = copy.deepcopy(indi)
            gbest_acc = copy.deepcopy(pbest_accSet[i])
            gbest_params = copy.deepcopy(pbest_params[i])
    return gbest, gbest_acc, gbest_params

def fitness_test(gbest_individual):
    filename = decode(gbest_individual, -1, -1)
    acc_set, num_parameters = fitnessEvaluate([filename], -1, is_test=True)
    return acc_set[0], num_parameters[0]

def evolveCNN(params):
    gen_no = 0
    Log.info('Initialize...')
    population = initialize_population(params)
    # Utils.save_population('pop', population, gen_no)

    Log.info('EVOLVE[%d-gen]-Begin evaluate the fitness' % (gen_no))
    acc_set, num_parameters = fitness_evaluate(population, gen_no, None)
    Log.info('EVOLVE[%d-gen]-Finish the evaluation' % (gen_no))

    # gbest
    [gbest_individual, gbest_acc, gbest_params]= update_best_individual(population, acc_set, num_parameters, gbest=None)
    Log.info('EVOLVE[%d-gen]-Finish the updating' % (gen_no))

    Utils.save_population_and_acc('population', population, acc_set, num_parameters, gen_no)
    # Utils.save_population_and_acc('pbest', pbest_individuals, pbest_accSet, gen_no)
    Utils.save_population_and_acc('gbest', [gbest_individual], [gbest_acc], [gbest_params], gen_no)

    gen_no += 1
    velocity_set = []
    for ii in range(len(population)):
        velocity_set.append([0]*len(population[ii]))

    for curr_gen in range(gen_no, params['num_iteration']):
        params['gen_no'] = curr_gen

        Log.info('EVOLVE[%d-gen]-Begin differential evolution' % (curr_gen))
        population, acc_set, num_parameters = evolve(population, acc_set, num_parameters, params)
        Log.info('EVOLVE[%d-gen]-Finish differential evolution' % (curr_gen))

        # Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness' % (curr_gen))
        # acc_set = fitness_evaluate(population, curr_gen)
        # Log.info('EVOLVE[%d-gen]-Finish the evaluation' % (curr_gen))

        [gbest_individual, gbest_acc, gbest_params] = update_best_individual(population, acc_set, num_parameters, gbest=[gbest_individual, gbest_acc, gbest_params])
        Log.info('EVOLVE[%d-gen]-Finish the updating' % (curr_gen))

        Utils.save_population_and_acc('population', population, acc_set, num_parameters, curr_gen)
        # Utils.save_population_and_acc('pbest', pbest_individuals, pbest_accSet, curr_gen)
        Utils.save_population_and_acc('gbest', [gbest_individual], [gbest_acc], [gbest_params], curr_gen)

    # final training and test on testset
    gbest_acc, num_parameters = fitness_test(gbest_individual)
    # num_parameters = Utils.calc_parameters_num(gbest_individual)
    Log.info('The acc of the best searched CNN architecture is [%.5f], number of parameters is [%d]' % (gbest_acc, num_parameters))
    Utils.save_population_and_acc('final_gbest', [gbest_individual], [gbest_acc], [num_parameters], -1)

if __name__ == '__main__':
    create_directory()
    params = Utils.get_init_params()
    evoCNN = evolveCNN(params)

