import random
import math
import architecture

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, AveragePooling2D, Layer

from typing import Union


def mutate(indiv, indpb, param_bounds):
    if indiv['model_struct'] == 'Random':
        indiv = mutate_architecture(indiv)
    indiv = gaussian_mutate_optimizer(indiv, indpb, param_bounds)
    return indiv,


def gaussian_mutate_optimizer(indiv, indpb, param_bounds):
    optimizer_dict = indiv['optimizer']
    optimizer_strat_dict = indiv['optimizer_strat']
    for key in optimizer_strat_dict:
        if random.random() < indpb:
            tau = (1.0 / (2 * (len(optimizer_dict) ** (1 / 2)))) ** (1 / 2)
            tau_prime = 1 / ((2 * (len(optimizer_dict))) ** (1 / 2))

            # mutating the strategy variable
            potential_step = optimizer_strat_dict[key] * math.exp(
                (tau_prime * random.gauss(0, 1))
                + (tau * random.gauss(0, 1))
            )
            optimizer_strat_dict[key] = keep_in_bounds(potential_step, param_bounds[key][2], param_bounds[key][3])

            if type(optimizer_dict[key]) == bool:
                if random.random() > optimizer_strat_dict[key]:  # a higher strategy value will mean less likely to flip
                    optimizer_dict[key] = not optimizer_dict[key]
            else:
                optimizer_dict[key] = keep_in_bounds(optimizer_dict[key] +
                                                     (optimizer_strat_dict[key] * random.gauss(0, 1)),
                                                     param_bounds[key][0],
                                                     param_bounds[key][1]
                                                     )
    indiv['optimizer'] = optimizer_dict
    indiv['optimizer_strat'] = optimizer_strat_dict
    return indiv


# TODO Decide on probabilities of each mutation
def mutate_architecture(indiv):
    try:
        model = Sequential.from_config(indiv['architecture'])
    except tensorflow.errors.ResourceExhaustedError:
        return indiv
    rand_val = random.random()

    # TODO double check that this works
    flatten_index = next(index for index, layer in enumerate(model.layers) if isinstance(layer, Flatten))

    if rand_val < 0.025:  # insert a random layer (2.5% chance)
        potential_model = insert_new_layer(model, flatten_index)
    elif rand_val < 0.05:  # insert a copied existing layer (2.5% chance)
        potential_model = insert_new_layer(model, flatten_index, is_copy=True)
    elif rand_val < 0.1:  # remove a random layer (5% chance)
        potential_model = remove_random_layer(model, flatten_index)
    elif rand_val < 0.125:  # move an existing layer (2.5% chance)
        potential_model = insert_new_layer(model, flatten_index, is_copy=True, remove_original=True)
    else:
        potential_model = model

    if potential_model is not None:
        del model  # Might not be necessary bc model is getting reassigned anyway
        tensorflow.keras.backend.clear_session()
        tensorflow.compat.v1.reset_default_graph()
        model = potential_model

    for i, layer in enumerate(model.layers[:-1]):  # Last layer (dense with 2 nodes) should never be mutated
        if i < flatten_index:
            input_shape = model.layers[i - 1].get_output_shape_at(0) if i > 0 else (None, 224, 224, 1)

            input_size = min(
                input_shape[1],
                input_shape[2]
            )
            if isinstance(layer, Conv2D):
                mutate_conv(layer, input_size)
            elif isinstance(layer, (MaxPooling2D, AveragePooling2D)):
                mutate_pool(layer, input_size)
        elif i > flatten_index:
            if isinstance(layer, Dense):
                mutate_dense(layer)
            elif isinstance(layer, Dropout):
                mutate_dropout(layer)

    indiv['architecture'] = model.get_config()
    del model
    tensorflow.keras.backend.clear_session()
    tensorflow.compat.v1.reset_default_graph()

    return indiv


def test_mutate_architecture():
    model = Sequential([
        Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 1)),  # input shape 1 or 3?
        AveragePooling2D(),
        Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
        AveragePooling2D(),
        Flatten(),
        Dense(units=60, activation='relu'),  # TODO change Units because we only have 2 classes to predict
        Dense(units=42, activation='relu'),  # TODO change Units because we only have 2 classes to predict
        Dense(units=1, activation='sigmoid')  # TODO change Units because we only have 2 classes to predict
    ])
    indiv = {'architecture': model.get_config()}
    print(indiv['architecture'])
    mutate_architecture(indiv)
    print(indiv['architecture'])


# !is_copy inserts random new layer randomly
# (is_copy, !remove_original) just creates a copy of random existing layer and inserts randomly
# (is_copy, remove_original) removes the original so that it effectively just moves the original somewhere else
def insert_new_layer(model: Sequential, flatten_index: int, is_copy=False, remove_original=False):
    layers = layer_list_from_config(model)
    insert_index = random.randint(1, len(layers) - 1)

    if insert_index <= flatten_index:
        input_size = min(
            model.layers[insert_index - 1].output_shape[1],
            model.layers[insert_index - 1].output_shape[2]
        )
        if is_copy:  # insert a copied layer
            try:
                orig_index = random.randint(1, flatten_index - 1)
            except ValueError:
                orig_index = 1
            layers.insert(insert_index, layers[orig_index])
            if remove_original:  # remove the original
                layers.pop(orig_index)
            elif input_size < 3:
                return model
        else:  # insert random new layer
            if input_size < 3:
                return model
            layers.insert(insert_index, random.choice([architecture.random_pool(input_size, "average"),
                                                       architecture.random_pool(input_size, "max"),
                                                       architecture.random_conv(input_size, False)
                                                       ]))
    else:
        if is_copy:  # insert a copied layer
            try:
                orig_index = random.randint(flatten_index + 1, len(layers)-1)
                layers.insert(insert_index, layers[orig_index])
                if remove_original:  # remove the original
                    layers.pop(orig_index)
            except:
                print(len(layers))
                print("orig_index", orig_index)
                print("insert_index", insert_index)
        else:  # insert random new layer
            layers.insert(insert_index, random.choice([architecture.random_dense(False),
                                                       architecture.random_dropout()
                                                       ]))

    del model
    tensorflow.keras.backend.clear_session()
    tensorflow.compat.v1.reset_default_graph()

    return reconstruct_model_from_layer_list(layers)


def remove_random_layer(model: Sequential, flatten_index: int):
    remove_index = random.randint(1, len(model.layers) - 2)
    if remove_index == flatten_index:  # can't remove flatten layer
        return model

    layers = layer_list_from_config(model)
    layers.pop(random.randint(1, len(model.layers) - 2))

    del model
    tensorflow.keras.backend.clear_session()
    tensorflow.compat.v1.reset_default_graph()

    return reconstruct_model_from_layer_list(layers)


# TODO Decide on probabilities of each mutation
def mutate_conv(conv_layer: Conv2D, input_size):
    if random.random() < 0.1:
        conv_layer.filters = keep_in_bounds(round(random.gauss(conv_layer.filters, 5)), 1, 500)

    if random.random() < 0.1:
        kernel_rowcol_size = keep_in_bounds(round(random.gauss(conv_layer.kernel_size[0], 5)), 2, input_size)
        conv_layer.kernel_size = (kernel_rowcol_size, kernel_rowcol_size)

    if random.random() < 0.05:
        strides_rowcol = keep_in_bounds(round(random.gauss(conv_layer.strides[0], 5)), 1, conv_layer.kernel_size[0])
        conv_layer.strides = (strides_rowcol, strides_rowcol)

    if random.random() < 0.05:
        if conv_layer.padding == 'valid':
            conv_layer.padding = 'same'
        else:
            conv_layer.padding = 'valid'


def mutate_pool(pool_layer: Union[MaxPooling2D, AveragePooling2D], input_size):
    if random.random() < 0.1:
        pool_rowcol_size =  keep_in_bounds(round(random.gauss(pool_layer.pool_size[0], 5)), 2, input_size)
        pool_layer.pool_size = (pool_rowcol_size, pool_rowcol_size)

    if random.random() < 0.05:
        strides_rowcol = keep_in_bounds(round(random.gauss(pool_layer.pool_size[0], 5)), 1, pool_layer.pool_size[0])
        pool_layer.strides = (strides_rowcol, strides_rowcol)

    if random.random() < 0.05:
        if pool_layer.padding == 'valid':
            pool_layer.padding = 'same'
        else:
            pool_layer.padding = 'valid'


def mutate_dense(dense_layer: Dense):
    if random.random() < 0.1:
        # Maybe should be dense_layer.units + round(random.gauss(0, 5))
        dense_layer.units = round(random.gauss(dense_layer.units, 20))


def mutate_dropout(dropout_layer: Dropout):
    if random.random() < 0.1:
        dropout_layer.rate = random.gauss(dropout_layer.rate, 0.1)


def keep_in_bounds(val, lower, upper):
    new_val = val
    if val < lower:
        new_val = lower
    elif val > upper:
        new_val = upper
    return new_val


def layer_list_from_config(model: Sequential):
    layers = []
    for layer_dict in model.get_config()['layers']:
        layer_type_str = layer_dict['class_name']
        if layer_type_str != 'InputLayer':
            layer = eval(layer_type_str).from_config(layer_dict['config'])
            layers.append(layer)

    del model
    tensorflow.keras.backend.clear_session()
    tensorflow.compat.v1.reset_default_graph()
    return layers


def reconstruct_model_from_layer_list(layers: list):
    model = Sequential()
    for layer in layers:
        try:
            model.add(layer)
        except ValueError:
            layer._name = layer.name + "_1"
        except tensorflow.errors.ResourceExhaustedError:
            print("Reconstruction failed after mutation, OOM.")
            return None
    return model


# test_mutate_architecture()
