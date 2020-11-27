import random
import math
import architecture

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, AveragePooling2D

from typing import Union


def gaussian_mutate_optimizer(indiv, indpb, param_bounds):
    for key in indiv:
        if random.random() < indpb:
            tau = (1.0 / (2 * (len(indiv) ** (1 / 2)))) ** (1 / 2)
            tau_prime = 1 / ((2 * (len(indiv))) ** (1 / 2))

            # mutating the strategy variable
            potential_step = indiv.strategy[key] * math.exp(
                (tau_prime * random.gauss(0, 1)) + (tau * random.gauss(0, 1)))
            if potential_step < param_bounds[key][2]:
                potential_step = param_bounds[key][2]
            elif potential_step > param_bounds[key][3]:
                potential_step = param_bounds[key][3]
            indiv.strategy[key] = potential_step

            if type(indiv[key]) == bool:
                if random.random() > indiv.strategy[key]:  # Thus, a higher strategy value will mean less likely to flip
                    indiv[key] = not indiv[key]
            else:
                potential = indiv[key] + (indiv.strategy[key] * random.gauss(0, 1))
                # TODO decide if value should be set to bounds, not mutated, or have a new value generated when the
                #  bounds are reached
                if potential < param_bounds[key][0]:
                    indiv[key] = param_bounds[key][0]
                elif potential > param_bounds[key][1]:
                    indiv[key] = param_bounds[key][1]
                else:
                    indiv[key] = potential

    return indiv,


# TODO Decide on probabilities of each mutation
def mutate_architecture(indiv):
    model: Sequential = indiv.architecture
    rand_val = random.random()

    # TODO double check that this works
    flatten_index = next(index for index, layer in enumerate(model.layers) if isinstance(layer, Flatten))

    if rand_val < 0.025:  # insert a random layer (2.5% chance)
        insert_new_layer(model, flatten_index)
    elif rand_val < 0.05:  # insert a copied existing layer (2.5% chance)
        insert_new_layer(model, flatten_index, is_copy=True)
    elif rand_val < 0.1:  # remove a random layer (5% chance)
        remove_random_layer(model, flatten_index)
    elif rand_val < 0.125:  # move an existing layer (2.5% chance)
        insert_new_layer(model, flatten_index, is_copy=True, remove_original=True)
    # TODO decide if there should be an 'else:' here, so layer parameters are only tweaked if none of the above happened
    for i, layer in enumerate(model.layers[-1]):  # Last layer (dense with 2 nodes) should never be mutated
        if i < flatten_index:
            if isinstance(layer, Conv2D):
                mutate_conv(layer)
            elif isinstance(layer, (MaxPooling2D, AveragePooling2D)):
                mutate_pool(layer)
        elif i > flatten_index:
            if isinstance(layer, Dense):
                mutate_dense(layer)
            elif isinstance(layer, Dropout):
                mutate_dropout(layer)


# !is_copy inserts random new layer randomly
# (is_copy, !remove_original) just creates a copy of random existing layer and inserts randomly
# (is_copy, remove_original) removes the original so that it effectively just moves the original somewhere else
def insert_new_layer(model: Sequential, flatten_index: int, is_copy=False, remove_original=False):
    insert_index = random.randint(1, model.layers.count() - 1)

    if insert_index <= flatten_index:
        input_size = min(
            model.layers[insert_index - 1].output_shape[1],
            model.layers[insert_index - 1].output_shape[2]
        )
        if is_copy:
            orig_index = random.randint(1, flatten_index)
            model.layers.insert(insert_index, model.layers.index(orig_index))
            if remove_original:
                model.layers.pop(orig_index)
        else:  # insert random new layer
            model.layers.insert(insert_index, random.choice([architecture.random_pool(input_size, "average"),
                                                             architecture.random_pool(input_size, "max"),
                                                             architecture.random_conv(input_size, False)
                                                             ]))
    else:
        if is_copy:
            orig_index = random.randint(flatten_index+1, model.layers.count())
            model.layers.insert(insert_index, model.layers.index(orig_index))
            if remove_original:
                model.layers.pop(orig_index)
        else:  # insert random new layer
            model.layers.insert(insert_index, random.choice([architecture.random_dense(False),
                                                             architecture.random_dropout()
                                                             ]))


def remove_random_layer(model: Sequential, flatten_index: int):
    remove_index = random.randint(1, model.layers.count() - 1)
    if remove_index == flatten_index:  # can't remove flatten layer
        return

    model.layers.pop(random.randint(1, model.layers.count() - 1))


# TODO Decide on probabilities of each mutation
def mutate_conv(conv_layer: Conv2D):
    if random.random() < 0.1:
        # Maybe should be conv_layer.filters + round(random.gauss(0, 5))
        conv_layer.filters = round(random.gauss(conv_layer.filters, 5))

    if random.random() < 0.1:
        kernel_rowcol_size = conv_layer.kernel_size[0] + random.choice([-1, 1])  # +- 1
        input_size = min(
            conv_layer.input_shape[1],
            conv_layer.input_shape[2]
        )
        if kernel_rowcol_size > input_size:
            kernel_rowcol_size = input_size
        elif kernel_rowcol_size < 2:
            kernel_rowcol_size = 2
        conv_layer.kernel_size = (kernel_rowcol_size, kernel_rowcol_size)

    if random.random() < 0.1:
        strides_rowcol = conv_layer.strides[0] + random.choice([-1, 1])  # +- 1
        if strides_rowcol > conv_layer.kernel_size[0]:
            strides_rowcol = conv_layer.kernel_size[0]
        elif strides_rowcol < 1:
            strides_rowcol = 1
        conv_layer.strides = (strides_rowcol, strides_rowcol)

    if random.random() < 0.05:
        if conv_layer.padding == 'valid':
            conv_layer.padding = 'same'
        else:
            conv_layer.padding = 'valid'


def mutate_pool(pool_layer: Union[MaxPooling2D, AveragePooling2D]):
    if random.random() < 0.1:
        pool_rowcol_size = pool_layer.pool_size[0] + random.choice([-1, 1])  # +- 1
        input_size = min(
            pool_layer.input_shape[1],
            pool_layer.input_shape[2]
        )
        if pool_rowcol_size > input_size:  # index might not be 2, need to test to find row/col size
            pool_rowcol_size = input_size
        elif pool_rowcol_size < 2:
            pool_rowcol_size = 2
        pool_layer.pool_size = (pool_rowcol_size, pool_rowcol_size)

    if random.random() < 0.1:
        strides_rowcol = pool_layer.strides[0] + random.choice([-1, 1])  # +- 1
        if strides_rowcol > pool_layer.pool_size[0]:
            strides_rowcol = pool_layer.pool_size[0]
        elif strides_rowcol < 1:
            strides_rowcol = 1
        pool_layer.strides = (strides_rowcol, strides_rowcol)

    if random.random() < 0.05:
        if pool_layer.padding == 'valid':
            pool_layer.padding = 'same'
        else:
            pool_layer.padding = 'valid'


def mutate_dense(dense_layer: Dense):
    if random.random() < 0.1:
        # Maybe should be conv_layer.filters + round(random.gauss(0, 5))
        dense_layer.filters = round(random.gauss(dense_layer.units, 20))


def mutate_dropout(dropout_layer: Dropout):
    if random.random() < 0.1:
        dropout_layer.rate = random.gauss(dropout_layer.rate, 0.1)
