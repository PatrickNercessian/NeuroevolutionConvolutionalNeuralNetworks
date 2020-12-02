from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
import random


def build_fn(model_struct):
    if model_struct == 'LeNet':
        layer_list = [
            Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(224, 224)),  # input shape 1 or 3?
            AveragePooling2D(),
            Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
            AveragePooling2D(),
            Flatten(),
            Dense(units=120, activation='relu'),  # TODO change Units because we only have 2 classes to predict
            Dense(units=84, activation='relu'),  # TODO change Units because we only have 2 classes to predict
            Dense(units=1, activation='softmax')  # TODO change Units because we only have 2 classes to predict
        ]
        return layer_list
    elif model_struct == 'AlexNet':
        layer_list = [
            Conv2D(filters=96, input_shape=(224, 224, 1), kernel_size=(11, 11), activation='relu', strides=(4, 4),
                   padding='valid'),  # Should input shape be 1 or 3?
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
            Conv2D(filters=256, kernel_size=(11, 11), activation='relu', strides=(1, 1), padding='valid'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
            Conv2D(filters=384, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='valid'),
            Conv2D(filters=384, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='valid'),
            Conv2D(filters=256, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='valid'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
            Flatten(),
            Dense(4096, input_shape=(224 * 224,), activation='relu'),
            Dropout(0.4),
            Dense(4096, activation='relu'),  # TODO change Units because we only have 2 classes to predict
            Dropout(0.4),
            Dense(1000, activation='relu'),  # TODO change Units because we only have 2 classes to predict
            Dropout(0.4),
            Dense(1, activation='softmax')  # TODO change Units because we only have 2 classes to predict
        ]
        return layer_list
    elif model_struct == 'Random':
        layer_list = [random_conv(224, True)]

        while True:
            input_size = min(
                layer_list[-1].output_shape[1],
                layer_list[-1].output_shape[2]
            )
            rand_val = random.random()
            if not any(isinstance(x, Flatten) for x in layer_list):  # if there's not a Flatten layer yet
                if rand_val < 0.3:
                    layer_list.append(random_conv(input_size, False))
                elif rand_val < 0.6:
                    layer_list.append(random_pool(input_size, 'max'))
                elif rand_val < 0.9:
                    layer_list.append(random_pool(input_size, 'average'))
                else:
                    layer_list.append(Flatten())
            else:  # if there is a Flatten layer already
                if rand_val < 0.35:
                    layer_list.append(random_dropout())
                elif rand_val < 0.7:
                    layer_list.append(random_dense(False))
                else:
                    break

        layer_list.append(random_dense(True))

    print("model was not LeNet, AlexNet, or Random")


def random_conv(input_size, is_first):
    filters = random.randint(1, 500)
    kernel_rowcol_size = random.randint(2, input_size)
    kernel_size = (kernel_rowcol_size, kernel_rowcol_size)

    strides = (random.randint(1, kernel_rowcol_size), random.randint(1, kernel_rowcol_size))

    padding = random.choice(["valid", "same"])

    if is_first:
        input_shape = (224, 224)  # Should input shape be 1 or 3?
        return Conv2D(input_shape=input_shape, filters=filters, kernel_size=kernel_size, strides=strides,
                      padding=padding, activation='relu')

    return Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu')


def random_pool(input_size, pool_type):
    pool_rowcol_size = random.randint(2, input_size)
    pool_size = (pool_rowcol_size, pool_rowcol_size)

    strides = (random.randint(1, pool_size[0]), random.randint(1, pool_size[1]))
    padding = random.choice(["valid", "same"])

    if pool_type == 'max':
        return MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)
    elif pool_type == 'average':
        return AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding)


def random_dropout():
    return Dropout(random.random())


def random_dense(is_last):
    units = 2 if is_last else random.randint(2, 500)
    activation = 'softmax' if is_last else 'relu'

    return Dense(units=units, activation=activation)
