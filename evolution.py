import math
import random

import numpy
from deap import base, algorithms
from deap import creator
from deap import tools

import models

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", dict, fitness=creator.FitnessMin, strategy=None)
creator.create("Strategy", dict)

toolbox = base.Toolbox()
opt_type = 'adam'

adam_param_bounds = {
    'lr': [0, 1, 0, 1], #[param min, param max, strategy min, strategy max]
    'lr_std': [0, 1, 0, 1],
    'decay_steps': [0, 100, 0, 1],
    'decay': [0, 1, 0, 1],
    'decay_std': [0, 1, 0, 1],
    'staircase': [True, False, 0, 1],
    'b1': [0, 1, 0, 1],
    'b1_std': [0, 1, 0, 1],
    'b2': [0, 1, 0, 1],
    'b2_std': [0, 1, 0, 1],
    'epsilon': [0, 1, 0, 1],
    'epsilon_std': [0, 1, 0, 1],
}

sgd_param_bounds = {
    'lr': [0, 1, 0, 1], #[param min, param max, strategy min, strategy max]
    'lr_std': [0, 1, 0, 1],
    'decay_steps': [0, 1, 0, 1],
    'decay': [0, 1, 0, 1],
    'decay_std': [0, 1, 0, 1],
    'staircase': [True, False, 0, 1],
    'momentum': [0, 1, 0, 1],
    'momentum_std': [0, 1, 0, 1],
    'nesterov': [True, False, 0, 1],
}

def build_param(opt_Type, is_strats):
    index = 0
    if is_strats:
        index = 2
    if opt_Type == 'adam':
        param_dict = {
            # to assign to the learning rate
            'lr': random.uniform(adam_param_bounds['lr'][index], adam_param_bounds['lr'][index+1]),
            'lr_std': random.uniform(adam_param_bounds['lr_std'][index], adam_param_bounds['lr_std'][index+1]),
            'decay_steps': random.uniform(adam_param_bounds['decay_steps'][index], adam_param_bounds['decay_steps'][index+1]),
            'decay': random.uniform(adam_param_bounds['decay'][index], adam_param_bounds['decay'][index+1]),
            'decay_std': random.uniform(adam_param_bounds['decay_std'][index], adam_param_bounds['decay_steps'][index+1]),
            'staircase': random.uniform(adam_param_bounds['staircase'][index], adam_param_bounds['staircase'][index+1]),
            'b1': random.uniform(adam_param_bounds['b1'][index], adam_param_bounds['b1'][index+1]),
            'b1_std': random.uniform(adam_param_bounds['b1_std'][index], adam_param_bounds['b1_std'][index+1]),
            'b2': random.uniform(adam_param_bounds['b2'][index], adam_param_bounds['b2'][index+1]),
            'b2_std': random.uniform(adam_param_bounds['b2_std'][index], adam_param_bounds['b2_std'][index+1]),
            'epsilon': random.uniform(adam_param_bounds['epsilon'][index], adam_param_bounds['epsilon'][index+1]),
            'epsilon_std': random.uniform(adam_param_bounds['epsilon_std'][index], adam_param_bounds['epsilon_std'][index+1]),
        }
        return param_dict
    elif opt_Type == 'sga':
        param_dict = {
            # to assign to the learning rate
            'lr': random.uniform(sgd_param_bounds['lr'][index], sgd_param_bounds['lr'][index+1]),
            'lr_std': random.uniform(sgd_param_bounds['lr_std'][index], sgd_param_bounds['lr_std'][index+1]),
            'decay_steps': random.uniform(sgd_param_bounds['decay_steps'][index], sgd_param_bounds['decay_steps'][index+1]),
            'decay': random.uniform(sgd_param_bounds['decay'][index], sgd_param_bounds['decay'][index+1]),
            'decay_std': random.uniform(sgd_param_bounds['decay_std'][index], sgd_param_bounds['decay_std'][index+1]),
            'staircase': random.uniform(sgd_param_bounds['staircase'][index], sgd_param_bounds['staircase'][index+1]),
            'momentum': random.uniform(sgd_param_bounds['momentum'][index], sgd_param_bounds['momentum'][index+1]),
            'momentum_std': random.uniform(sgd_param_bounds['momentum_std'][index], sgd_param_bounds['momentum_std'][index+1]),
            'nesterov': random.uniform(sgd_param_bounds['nesterov'][index], sgd_param_bounds['nesterov'][index+1]),
        }
        return param_dict
    print("optType was not Adam or SGD")

def generate_indiv_adam(indiv_class, strat_class, opt_type):
    obj_dict = build_param(opt_type, False)
    indiv = indiv_class(obj_dict)
    indiv.strategy = strat_class(build_param(opt_type, True))
    return indiv


def avg_crossover(indiv1, indiv2):
    print()
    # Use list of keys to iterate and avg floats/ints and select booleans randomly


def select_crossover(indiv1, indiv2):
    print("should select from one of the parents randomly")

def gaussian_mutate(indiv, indpb, param_bounds):
    for index, key in enumerate(indiv):
        if random.random() < indpb:
            tau = (1.0 / (2 * (len(indiv) ** (1/2)))) ** (1/2)
            tau_prime = 1 / ((2 * (len(indiv))) ** (1/2))

            potential_step = indiv.strategy[key] * math.exp((tau_prime * random.gauss(0, 1)) + (tau * random.gauss(0, 1)))

            if(potential_step < param_bounds[key][2]):
                potential_step = param_bounds[key][2]
            elif(potential_step > param_bounds[key][3]):
                potential_step = param_bounds[key][3]

            indiv.strategy[key] = potential_step

            if(type(indiv[key]) == bool):
                #TODO make it so the strategy value impacts whether the value flips to true/false
                print("", end = "")
            else:
                potential = indiv[key] + (indiv.strategy[key] * random.gauss(0, 1))
                # TODO decide if value should be set to bounds, not mutated, or have a new value generated when the bounds are reached
                if potential < param_bounds[key][0]:
                    indiv[key] = param_bounds[key][0]
                elif potential > param_bounds[key][1]:
                    indiv[key] = param_bounds[key][1]
                else:
                    indiv[key] = potential

    return indiv,


def fitness(indiv):
    # Fitness should run the model and return the accuracy
    return ();


if opt_type == 'adam':
    toolbox.register("individual", generate_indiv_adam, creator.Individual, creator.Strategy, opt_type)
    toolbox.register("mutate", gaussian_mutate, indpb=1.0, param_bounds=adam_param_bounds)
else:
    toolbox.register("individual", generate_indiv_adam, creator.Individual, creator.Strategy, opt_type)
    toolbox.register("mutate", gaussian_mutate, indpb=1.0, param_bounds=sgd_param_bounds)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("avg_crossover", avg_crossover)
toolbox.register("select_crossover", select_crossover)

toolbox.register("select", tools.selBest)
toolbox.register("evaluate", fitness)


def run():
    MU, LAMBDA = 5, 35
    pop_q1 = toolbox.population(n=MU)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)

    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, logbook = algorithms.eaMuPlusLambda(pop_q1, toolbox, mu=MU, lambda_=LAMBDA,
                                             cxpb=0.5, mutpb=0.5, ngen=14, stats=stats, halloffame=hof, verbose=False)
    print(hof.items[0].fitness, hof.items[0])
    return pop, logbook, hof


run()
