import math
import random

import numpy
from deap import base, algorithms
from deap import creator
from deap import tools

import models

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", dict, fitness=creator.FitnessMin, strategy=None)

toolbox = base.Toolbox()


def generate_indiv_adam(indiv_class):
    vars = models.build_param('Adam')
    indiv = indiv_class(vars)
    return indiv


def avg_crossover(indiv1, indiv2):
    print()
    # Use list of keys to iterate and avg floats/ints and select booleans randomly


def select_crossover(indiv1, indiv2):
    print("should select from one of the parents randomly")


def gaussian_mutate(indiv, indpb):
    # for i in enumerate(indiv):
    #     if random.random() < indpb:
    #         potential = indiv[i[0]] + random.gauss(0, indiv.strategy)
    #         if i[0] == 0:
    #             if 40 >= potential >= -60:
    #                 indiv[i[0]] = potential
    #         if i[0] == 1:
    #             if 70 >= potential >= -30:
    #                 indiv[i[0]] = potential
    #
    # return indiv,


def fitness(indiv):
    # Fitness should run the model and return the accuracy
    print("")


toolbox.register("individual", generate_indiv_adam, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("avg_crossover", avg_crossover)
toolbox.register("select_crossover", select_crossover)

toolbox.register("select", tools.selBest)
toolbox.register("mutate", gaussian_mutate, indpb=1.0)
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
