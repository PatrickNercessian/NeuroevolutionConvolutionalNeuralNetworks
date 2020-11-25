import random
import math

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D

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


def mutate_architecture(indiv):
    # model = indiv.architecture
    model = Sequential()

    for layer in model.layers[-1]:  # Last layer (dense with 2 nodes) should never be mutated
        if isinstance(layer, Conv2D):
            print("placeholder")


# TODO Decide on probabilities of each mutation
def mutate_conv(conv_layer: Conv2D):
    if random.random() < 0.1:
        # Maybe should be conv_layer.filters + round(random.gauss(0, 5))
        conv_layer.filters = round(random.gauss(conv_layer.filters, 5))

    if random.random() < 0.1:
        kernel_rowcol_size = conv_layer.kernel_size[0] + random.choice([-1, 1])  # +- 1
        if kernel_rowcol_size > conv_layer.input_shape[2]:  # index might not be 2, need to test to find row/col size
            kernel_rowcol_size = conv_layer.input_shape[2]
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
        if pool_rowcol_size > pool_layer.input_shape[2]:  # index might not be 2, need to test to find row/col size
            pool_rowcol_size = pool_layer.input_shape[2]
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
