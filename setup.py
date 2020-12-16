import random
import numpy
import pandas as pd
from deap import base, algorithms
from deap import creator
from deap import tools
import matplotlib.pyplot as plt
import architecture
import mutation
import recombination
import time

from PIL import Image, ImageOps
from numpy import asarray
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow
from tensorflow.keras import Sequential

tf_config = tensorflow.compat.v1.ConfigProto(
    gpu_options=tensorflow.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.12))
tf_config.gpu_options.allow_growth = True
session = tensorflow.compat.v1.Session(config=tf_config)
tensorflow.compat.v1.keras.backend.set_session(session)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", dict, fitness=creator.FitnessMax, strategy=None)
creator.create("Strategy", dict)

toolbox = base.Toolbox()

adam_param_bounds = {
    'lr': [0, 1, 0, 1],  # [param min, param max, strategy min, strategy max]
    'decay_steps': [0, 100, 0, 100],
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
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.4, random_state=3)
    return np.array(x_train).reshape(-1, 224, 224, 1), np.array(x_test).reshape(-1, 224, 224, 1), np.array(
        y_train), np.array(y_test)


def convert_images():
    directory = 'archive/brain_tumor_dataset/no/'
    list_nparrays = []
    list_labels = []

    for i, filename in enumerate(os.listdir(directory)):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image = Image.open(directory + filename)
            image_resized_grayscale = ImageOps.grayscale(image.resize((224, 224)))
            data = asarray(image_resized_grayscale)
            list_nparrays.append(data)
            list_labels.append(0)

    directory = 'archive/brain_tumor_dataset/yes/'
    for i, filename in enumerate(os.listdir(directory)):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image = Image.open(directory + filename)
            image_resized_grayscale = ImageOps.grayscale(image.resize((224, 224)))
            data = asarray(image_resized_grayscale)
            list_nparrays.append(data)
            list_labels.append(1)

    return list_nparrays, list_labels


def build_param(optimizer, is_strats):
    index = 2 if is_strats else 0
    if optimizer == 'adam':
        b1 = random.uniform(adam_param_bounds['b1'][index], adam_param_bounds['b1'][index + 1])
        param_dict = {
            # to assign to the learning rate
            'lr': random.uniform(adam_param_bounds['lr'][index], adam_param_bounds['lr'][index + 1]),
            'decay_steps': random.randint(adam_param_bounds['decay_steps'][index],
                                          adam_param_bounds['decay_steps'][index + 1]),
            'decay': random.uniform(adam_param_bounds['decay'][index], adam_param_bounds['decay'][index + 1]),
            'staircase': random.choice([adam_param_bounds['staircase'][index],
                                        adam_param_bounds['staircase'][index + 1]]),
            'b1': b1,
            'b2': random.uniform(b1, adam_param_bounds['b2'][index + 1]),
            'epsilon': random.uniform(adam_param_bounds['epsilon'][index], adam_param_bounds['epsilon'][index + 1])
        }
        if not is_strats:
            param_dict.update({'optimizer_type': optimizer})
        else:
            param_dict.update({'staircase': random.uniform(adam_param_bounds['staircase'][index],
                                                           adam_param_bounds['staircase'][index + 1])
                               })
        return param_dict
    elif optimizer == 'sga':
        param_dict = {
            # to assign to the learning rate
            'lr': random.uniform(sgd_param_bounds['lr'][index], sgd_param_bounds['lr'][index + 1]),
            'decay_steps': random.randint(sgd_param_bounds['decay_steps'][index],
                                          sgd_param_bounds['decay_steps'][index + 1]),
            'decay': random.uniform(sgd_param_bounds['decay'][index], sgd_param_bounds['decay'][index + 1]),
            'staircase': random.choice([sgd_param_bounds['staircase'][index],
                                        sgd_param_bounds['staircase'][index + 1]]),
            'momentum': random.uniform(sgd_param_bounds['momentum'][index], sgd_param_bounds['momentum'][index + 1]),
            'nesterov': random.choice([sgd_param_bounds['nesterov'][index],
                                       sgd_param_bounds['nesterov'][index + 1]])
        }
        if not is_strats:
            param_dict.update({'optimizer_type': optimizer})
        else:
            param_dict.update({'staircase': random.uniform(sgd_param_bounds['staircase'][index],
                                                           sgd_param_bounds['staircase'][index + 1])
                               })
            param_dict.update({'staircase': random.uniform(sgd_param_bounds['staircase'][index],
                                                           sgd_param_bounds['staircase'][index + 1])
                               })

        return param_dict
    print("optType was not Adam or SGD")


def generate_indiv(indiv_class, strat_class, optimizer, model_struct):
    indiv = indiv_class()

    indiv['model_struct'] = model_struct
    indiv['optimizer'] = indiv_class(build_param(optimizer, False))
    indiv['optimizer_strat'] = strat_class(build_param(optimizer, True))
    indiv['architecture'] = architecture.build_fn(model_struct)
    # TODO Maybe add indiv['architecture_strat']
    return indiv


def fitness(indiv):
    optimizer_dict = indiv['optimizer']
    # if possible get computation time and add penalty
    if optimizer_dict['optimizer_type'] == 'adam':
        opt = tensorflow.keras.optimizers.Adam(learning_rate=optimizer_dict['lr'], beta_1=optimizer_dict['b1'],
                                               beta_2=optimizer_dict['b2'],
                                               epsilon=optimizer_dict['epsilon'])
    elif optimizer_dict['optimizer_type'] == 'sga':
        opt = tensorflow.keras.optimizers.SGD(learning_rate=optimizer_dict['lr'], momentum=optimizer_dict['momentum'],
                                              nesterov=optimizer_dict['nesterov'])

    try:
        model = Sequential.from_config(indiv['architecture'])
        num_trainable_params = np.sum([tensorflow.keras.backend.count_params(w) for w in model.trainable_weights])
        if num_trainable_params > 10000000:
            fit = 0
        else:
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.fit(data_split[0], data_split[2], batch_size=32, epochs=20, verbose=False)

            fit = model.evaluate(data_split[1], data_split[3])[1]
    except tensorflow.errors.ResourceExhaustedError:
        print("Went OOM with architecture", indiv['architecture'])
        fit = 0
    except Exception as e:
        print(e)
        fit = 0

    try:
        del model
    except UnboundLocalError as e:
        print(e)
    tensorflow.keras.backend.clear_session()
    tensorflow.compat.v1.reset_default_graph()

    return (fit,)


img_data, img_labels = convert_images()
data_split = split(img_data, img_labels)


def setup_toolbox(optimizer, model_struct):
    toolbox.register("individual", generate_indiv, creator.Individual, creator.Strategy, optimizer, model_struct)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", recombination.optimizer_crossover)
    if optimizer == 'adam':
        toolbox.register("mutate", mutation.mutate, indpb=1.0, param_bounds=adam_param_bounds)
    else:
        toolbox.register("mutate", mutation.mutate, indpb=1.0, param_bounds=sgd_param_bounds)

    toolbox.register("select", tools.selBest)
    toolbox.register("evaluate", fitness)


def run(optimizer, model_struct):
    setup_toolbox(optimizer, model_struct)

    MU, LAMBDA = 4, 12
    population = toolbox.population(n=MU)

    for indiv in population:
        print(indiv['optimizer'])
        print(indiv['architecture'])

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)

    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, logbook = algorithms.eaMuPlusLambda(population, toolbox, mu=MU, lambda_=LAMBDA,
                                             cxpb=0.5, mutpb=0.5, ngen=15, stats=stats, halloffame=hof, verbose=True)
    logbook.header = "gen", "avg", "max"
    print("Best Individual had fitness of", hof.items[0].fitness)
    print("with optimizer", hof.items[0]['optimizer'])
    print("with optimizer strategy varibles", hof.items[0]['optimizer_strat'])

    if model_struct == "Random":
        print("with architecture:")
        for layer_dict in hof.items[0]['architecture']['layers']:
            print(layer_dict)

    return pop, logbook, hof


# run("adam", "LeNet")

runs = 50
x = 0
file_names = []
while x < 50:
    file_names.append(x)
    x += 1

x = 0

hall_of_fame = []

while x < runs:
    pop, logbook, hof = run("adam", "LeNet")
    gen = logbook.select("gen")
    fit_max = logbook.select("max")
    # plt.plot(gen, fit_max, label='Best Fitness in each Generation')
    # plt.xlabel('Generation')
    # plt.ylabel('Fitness')
    hall_of_fame.append(hof.items[0])
    df_log = pd.DataFrame(logbook)
    df_log.to_csv('./CSVs/{}.csv'.format(file_names[x]))
    x += 1

df_log = pd.DataFrame(hall_of_fame)
df_log.to_csv('./CSVs/Results/HoF.csv')
