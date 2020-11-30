import random
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, AveragePooling2D


def avg_crossover(indiv1, indiv2):
    new_indiv = dict()
    for key in indiv1:
        if type(indiv1[key] == bool):
            new_indiv[key] = indiv1[key] if random.random() < 0.5 else indiv2[key]  # randomly selects one
        else:
            new_indiv[key] = (indiv1[key] + indiv2[key]) / 2
    return new_indiv


def select_crossover(indiv1, indiv2):
    new_indiv1, new_indiv2 = dict(), dict()
    for key in indiv1:
        if random.random() < 0.5:
            new_indiv1[key] = indiv1[key]
            new_indiv2[key] = indiv2[key]
        else:
            new_indiv1[key] = indiv2[key]
            new_indiv2[key] = indiv1[key]


def architecture_crossover(indiv1, indiv2):
    new_model = Sequential()
    model1: Sequential = indiv1.architecture
    model2: Sequential = indiv2.architecture
    i1, i2 = 1, 1  # Ignore the first layer

    # TODO likely that this will not work as is because output shapes will change in new_model, and
    #  we are simply copy-pasting layers with their existing input shapes
    while i1 < model1.layers.count() and i2 < model2.layers.count():
        layer1, layer2 = model1.layers.index(i1), model2.layers.index(i2)

        # When reaching end of one (or both) loops
        if i1 == model1.layers.count() - 1:
            if random.random() < 0.5:  # Add the final Dense layer and break out of loop
                new_model.add(layer1)
                break
            else:  # Add the other layer and increment, but continue to next loop run (doesn't increment i1)
                new_model.add(layer2)
                i2 += 1; continue  # Note: even if both layers are final Dense, this would still cause loop to end
        elif i2 == model2.layers.count() - 1:
            if random.random() < 0.5:  # Add the final Dense layer and break out of loop
                new_model.add(layer2)
                break
            else:  # Add the other layer and increment, but continue to next loop run (doesn't increment i2)
                new_model.add(layer1)
                i1 += 1; continue  # Note: even if both layers are final Dense, this would still cause loop to end

        if not isinstance(layer1, type(layer2)):  # If different layer types
            if isinstance(layer1, Flatten):
                if random.random() < 0.5:  # TODO Might not be best; will favor shorter children bc 50% chance of ending
                    new_model.add(Flatten)
                    while not isinstance(model2.layers.index(i2), Flatten):  # increment i2 until reaching Flatten layer
                        i2 += 1
                else:
                    new_model.add(layer2)
                    i2 += 1; continue # Don't want i1 to increment, because then it'd move onto Dense/Dropout sections
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
                conv_results = crossover_convolutional(layer1, layer2)
                new_model.add(Conv2D(filters=conv_results[0], kernel_size=(conv_results[1], conv_results[1]),
                                     strides=(conv_results[2], conv_results[2]), padding=conv_results[3]
                                     ))
            elif all(isinstance(x, (MaxPooling2D, AveragePooling2D)) for x in (layer1, layer2)):  # If both are pooling
                pool_results = crossover_pooling(layer1, layer2)
                layer_type = random.choice([type(layer1), type(layer2)])
                new_model.add(layer_type(pool_size=(pool_results[0], pool_results[0]),
                                         strides=(pool_results[1], pool_results[1]),
                                         padding=pool_results[2]
                                         ))
            elif all(isinstance(x, Dense) for x in (layer1, layer2)):  # If both are Dense
                new_model.add(Dense(round((layer1.units + layer2.units) / 2)))
            elif all(isinstance(x, Dropout) for x in (layer1, layer2)):  # If both are Dropout
                new_model.add(Dropout((layer1.rate + layer2.rate) / 2))
            else:
                print("An error/edge case has occurred: no recombination possible for this layer")

        i1 += 1
        i2 += 1


def crossover_convolutional(conv1: Conv2D, conv2: Conv2D):
    filters = round((conv1.filters + conv2.filters) / 2)
    kernel_rowcol_size = round((conv1.kernel_size[0] + conv2.kernel_size[0]) / 2)
    strides_size = round((conv1.strides[0] + conv2.strides[0]) / 2)
    padding = random.choice([conv1.padding, conv2.padding])
    return filters, kernel_rowcol_size, strides_size, padding


def crossover_pooling(pool1, pool2):
    pool_rowcol_size = round((pool1.pool_size[0] + pool2.pool_size[0]) / 2)
    strides_size = round((pool1.strides[0] + pool2.strides[0]) / 2)
    padding = random.choice([pool1.padding, pool2.padding])
    return pool_rowcol_size, strides_size, padding
