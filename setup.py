import random
import numpy
import tensorflow
from tensorflow.keras import Sequential
from deap import base, algorithms
from deap import creator
from deap import tools

import architecture
import mutation
import recombination

from PIL import Image, ImageOps
from numpy import asarray
import numpy as np
import os

creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", dict, fitness=creator.FitnessMax, strategy=None)
creator.create("Strategy", dict)

toolbox = base.Toolbox()

adam_param_bounds = {
    'lr': [0, 1, 0, 1],  # [param min, param max, strategy min, strategy max]
    'decay_steps': [0, 100, 0, 1],
    'decay': [0, 1, 0, 1],
    'staircase': [True, False, 0, 1],
    'b1': [0, 1, 0, 1],
    'b2': [0, 1, 0, 1],
    'epsilon': [0, 1, 0, 1]
}

sgd_param_bounds = {
    'lr': [0, 1, 0, 1],  # [param min, param max, strategy min, strategy max]
    'decay_steps': [0, 1, 0, 1],
    'decay': [0, 1, 0, 1],
    'staircase': [True, False, 0, 1],
    'momentum': [0, 1, 0, 1],
    'nesterov': [True, False, 0, 1],
}


def split(images, labels):
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=3)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


def convert_images():
    directory = 'archive/brain_tumor_dataset/no/'
    list_nparrays = []
    list_labels = []
    #list_labelsnoTumor = []
    #this could help accuracy but we must also change the output shape in architecture to this manner
    for i, filename in enumerate(os.listdir(directory)):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image = Image.open(directory + filename)
            image_resized_grayscale = ImageOps.grayscale(image.resize((224, 224)))
            data = asarray(image_resized_grayscale)
            list_nparrays.append(data)
            list_labels.append(0)
            # list_labelsnoTumor.append(1)

    directory = 'archive/brain_tumor_dataset/yes/'
    for i, filename in enumerate(os.listdir(directory)):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image = Image.open(directory + filename)
            image_resized_grayscale = ImageOps.grayscale(image.resize((224, 224)))
            data = asarray(image_resized_grayscale)
            list_nparrays.append(data)
            list_labels.append(1)
            # list_labelsnoTumor.append(0)

    return list_nparrays, list_labels


def build_param(optimizer, is_strats):
    index = 2 if is_strats else 0
    if optimizer == 'adam':
        param_dict = {
            # to assign to the learning rate
            'optimizer': optimizer,
            'lr': random.uniform(adam_param_bounds['lr'][index], adam_param_bounds['lr'][index + 1]),
            'decay_steps': random.uniform(adam_param_bounds['decay_steps'][index],  # TODO needs to be random.randint()
                                          adam_param_bounds['decay_steps'][index + 1]),
            'decay': random.uniform(adam_param_bounds['decay'][index], adam_param_bounds['decay'][index + 1]),
            'staircase': random.uniform(adam_param_bounds['staircase'][index],
                                        adam_param_bounds['staircase'][index + 1]),
            'b1': random.uniform(adam_param_bounds['b1'][index], adam_param_bounds['b1'][index + 1]),
            'b2': random.uniform(adam_param_bounds['b2'][index], adam_param_bounds['b2'][index + 1]),
            'epsilon': random.uniform(adam_param_bounds['epsilon'][index], adam_param_bounds['epsilon'][index + 1]),
        }
        return param_dict
    elif optimizer == 'sga':
        param_dict = {
            # to assign to the learning rate
            'optimizer': optimizer,
            'lr': random.uniform(sgd_param_bounds['lr'][index], sgd_param_bounds['lr'][index + 1]),
            'decay_steps': random.uniform(sgd_param_bounds['decay_steps'][index],
                                          sgd_param_bounds['decay_steps'][index + 1]),
            'decay': random.uniform(sgd_param_bounds['decay'][index], sgd_param_bounds['decay'][index + 1]),
            'staircase': random.uniform(sgd_param_bounds['staircase'][index], sgd_param_bounds['staircase'][index + 1]),
            'momentum': random.uniform(sgd_param_bounds['momentum'][index], sgd_param_bounds['momentum'][index + 1]),
            'nesterov': random.uniform(sgd_param_bounds['nesterov'][index], sgd_param_bounds['nesterov'][index + 1]),
        }
        return param_dict
    print("optType was not Adam or SGD")


def generate_indiv(indiv_class, strat_class, optimizer, model_struct):
    indiv = indiv_class(build_param(optimizer, False))
    indiv.strategy = strat_class(build_param(optimizer, True))
    indiv.architecture = architecture.build_fn(model_struct)
    # TODO Maybe add indiv.architecture.strategy?
    return indiv


def fitness(indiv):
    model = Sequential(indiv.architecture)
    # if possible get computation time and add penalty
    if indiv['optimizer'] == 'adam':
        lr_schedule = tensorflow.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=indiv['lr'],
                                                                             decay_steps=indiv['decay_steps'],
                                                                             decay_rate=indiv['decay'],
                                                                             staircase=indiv['staircase'])
        opt = tensorflow.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=indiv['b1'], beta_2=indiv['b2'],
                                               epsilon=indiv['epsilon'])
    elif indiv['optimizer'] == 'sga':
        lr_schedule = tensorflow.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=indiv['lr'],
                                                                             decay_steps=indiv['decay_steps'],
                                                                             decay_rate=indiv['decay'],
                                                                             staircase=indiv['staircase'])
        opt = tensorflow.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=indiv['momentum'],
                                              nesterov=indiv['nesterov'])
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    x_train = data_split[0].reshape(-1, 224, 224, 1)
    x_test = data_split[1].reshape(-1, 224, 224, 1)
    # print(model.summary())
    model.fit(x_train, data_split[2], batch_size=20, epochs=2)
    fit = model.evaluate(x_test, data_split[3])[1]
    print(fit)
    return fit,


img_data, img_labels = convert_images()
data_split = split(img_data, img_labels)


def setup_toolbox(optimizer, model_struct):
    toolbox.register("individual", generate_indiv, creator.Individual, creator.Strategy, optimizer, model_struct)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("avg_crossover", recombination.avg_crossover)
    toolbox.register("select_crossover", recombination.select_crossover)
    if optimizer == 'adam':
        toolbox.register("mutate", mutation.gaussian_mutate_optimizer, indpb=1.0, param_bounds=adam_param_bounds)
    else:
        toolbox.register("mutate", mutation.gaussian_mutate_optimizer, indpb=1.0, param_bounds=sgd_param_bounds)

    toolbox.register("select", tools.selBest)
    toolbox.register("evaluate", fitness)


def run(optimizer, model_struct):
    setup_toolbox(optimizer, model_struct)

    MU, LAMBDA = 3, 3
    population = toolbox.population(n=MU)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)

    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, logbook = algorithms.eaMuPlusLambda(population, toolbox, mu=MU, lambda_=LAMBDA,
                                             cxpb=0.5, mutpb=0.5, ngen=14, stats=stats, verbose=True)

    # pop, logbook = algorithms.eaMuPlusLambda(population, toolbox, mu=MU, lambda_=LAMBDA,
    #                                          cxpb=0.5, mutpb=0.5, ngen=14, stats=stats, halloffame=hof, verbose=False)
    # print(hof.items[0].fitness, hof.items[0])
    return pop, logbook, hof


run("adam", "AlexNet")
