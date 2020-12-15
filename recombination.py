import random
from typing import Union

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, AveragePooling2D


def optimizer_crossover(indiv1, indiv2):
    if random.random() < 0.5:
        return weighted_avg_crossover(indiv1, indiv2)
    else:
        return select_crossover(indiv1, indiv2)


def weighted_avg_crossover(indiv1, indiv2):
    for key in indiv1['optimizer']:
        rand_val = random.random()
        if type(indiv1['optimizer'][key]) == bool:
            indiv1['optimizer'][key] = indiv1['optimizer'][key] if rand_val < 0.5 else indiv2['optimizer'][key]  # randomly selects one
            indiv2['optimizer'][key] = indiv2['optimizer'][key] if rand_val < 0.5 else indiv1['optimizer'][key]  # randomly selects one
        elif key != 'optimizer_type':
            indiv1['optimizer'][key] = ((rand_val * indiv1['optimizer'][key])
                                        + ((1 - rand_val) * indiv2['optimizer'][key])) / 2.0  # random weight to each
            indiv2['optimizer'][key] = (((1 - rand_val) * indiv1['optimizer'][key])
                                        + (rand_val * indiv2['optimizer'][key])) / 2.0  # random weight to each
    return indiv1, indiv2


def select_crossover(indiv1, indiv2):
    for key in indiv1['optimizer']:
        if random.random() > 0.5:
            temp = indiv1['optimizer'][key]
            indiv1['optimizer'][key] = indiv2['optimizer'][key]
            indiv2['optimizer'][key] = temp
    return indiv1, indiv2


def architecture_crossover(indiv1, indiv2):
    new_model = Sequential()
    model1, model2 = Sequential.from_config(indiv1['architecture']), Sequential.from_config(indiv2['architecture'])
    i1, i2 = 1, 1  # Ignore the first layer

    while i1 < len(model1.layers) and i2 < len(model2.layers):
        layer1, layer2 = model1.layers.index(i1), model2.layers.index(i2)

        # When reaching end of one (or both) loops
        if i1 == len(model1.layers) - 1:
            if random.random() < 0.5:  # Add the final Dense layer and break out of loop
                new_model.add(layer1)
                break
            else:  # Add the other layer and increment, but continue to next loop run (doesn't increment i1)
                new_model.add(layer2)
                i2 += 1; continue  # Note: even if both layers are final Dense, this should still cause loop to end
        elif i2 == len(model2.layers) - 1:
            if random.random() < 0.5:  # Add the final Dense layer and break out of loop
                new_model.add(layer2)
                break
            else:  # Add the other layer and increment, but continue to next loop run (doesn't increment i2)
                new_model.add(layer1)
                i1 += 1; continue  # Note: even if both layers are final Dense, this should still cause loop to end

        # TODO maybe will not work as is because output shapes will change in new_model, and
        #  we are simply copy-pasting layers with their existing input shapes.
        #  Potential solution would be to recreate the layer with the same parameter
        if not isinstance(layer1, type(layer2)):  # If different layer types
            if isinstance(layer1, Flatten):
                if random.random() < 0.5:  # TODO Might not be best; will favor shorter children bc 50% chance of ending
                    new_model.add(Flatten)
                    while not isinstance(model2.layers.index(i2), Flatten):  # increment i2 until reaching Flatten layer
                        i2 += 1
                else:
                    new_model.add(layer2)
                    i2 += 1; continue  # Don't want i1 to increment, because then it'd move onto Dense/Dropout sections
            elif isinstance(layer2, Flatten):
                if random.random() < 0.5:  # TODO Might not be best; will favor shorter children bc 50% chance of ending
                    new_model.add(Flatten)
                    while not isinstance(model1.layers.index(i1), Flatten):  # increment i1 until reaching Flatten layer
                        i1 += 1
                else:
                    new_model.add(layer1)
                    i1 += 1; continue  # Don't want i2 to increment, because then it'd move onto Dense/Dropout sections
            else:
                new_model.add(random.choice([layer1, layer2]))
        else:  # If same layer types
            # TODO to save computation, only need to check for one of the layers (because they are both the same)
            if all(isinstance(x, Conv2D) for x in (layer1, layer2)):  # If both are convolutional
                new_model.add(crossover_convolutional(layer1, layer2))
            elif all(isinstance(x, (MaxPooling2D, AveragePooling2D)) for x in (layer1, layer2)):  # If both are pooling
                layer_type = random.choice([type(layer1), type(layer2)])
                new_model.add(crossover_pooling(layer_type, layer1, layer2))
            elif all(isinstance(x, Dense) for x in (layer1, layer2)):  # If both are Dense
                new_model.add(Dense(round((layer1.units + layer2.units) / 2)))
            elif all(isinstance(x, Dropout) for x in (layer1, layer2)):  # If both are Dropout
                new_model.add(Dropout((layer1.rate + layer2.rate) / 2))
            elif all(isinstance(x, Flatten) for x in (layer1, layer2)):
                new_model.add(Flatten())
            else:
                print("An unexpected edge case has occurred: no recombination possible for this layer")

        i1 += 1
        i2 += 1

        return new_model


def crossover_convolutional(conv1: Conv2D, conv2: Conv2D):
    filters = round((conv1.filters + conv2.filters) / 2)
    kernel_rowcol_size = round((conv1.kernel_size[0] + conv2.kernel_size[0]) / 2)
    strides_size = round((conv1.strides[0] + conv2.strides[0]) / 2)
    padding = random.choice([conv1.padding, conv2.padding])
    return Conv2D(filters=filters, kernel_size=(kernel_rowcol_size, kernel_rowcol_size),
                  strides=(strides_size, strides_size), padding=padding)


def crossover_pooling(layer_type: Union[MaxPooling2D, AveragePooling2D], pool1: Union[MaxPooling2D, AveragePooling2D],
                      pool2: Union[MaxPooling2D, AveragePooling2D]):
    pool_rowcol_size = round((pool1.pool_size[0] + pool2.pool_size[0]) / 2)
    strides_size = round((pool1.strides[0] + pool2.strides[0]) / 2)
    padding = random.choice([pool1.padding, pool2.padding])
    return layer_type(pool_size=(pool_rowcol_size, pool_rowcol_size), strides=(strides_size, strides_size),
                      padding=padding)
