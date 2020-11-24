import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import random

def build_fn(model_struct):
    if model_struct == 'LeNet':
        model = tf.keras.Sequential([
            layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 1)),
            layers.AveragePooling2D(),
            layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
            layers.AveragePooling2D(),
            layers.Flatten(),
            layers.Dense(units=120, activation='relu'),
            layers.Dense(units=84, activation='relu'),
            layers.Dense(units=10, activation='softmax')
        ])
        return model
    elif model_struct == 'AlexNet':
        model = tf.keras.Sequential([
            Conv2D(filters=96, input_shape=(224, 224, 3), kernel_size=(11, 11), activation='relu', strides=(4, 4),
                   padding='valid'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
            Conv2D(filters=256, kernel_size=(11, 11), activation='relu', strides=(1, 1), padding='valid'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
            Conv2D(filters=384, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='valid'),
            Conv2D(filters=384, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='valid'),
            Conv2D(filters=256, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='valid'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
            Flatten(),
            Dense(4096, input_shape=(224 * 224 * 3,), activation='relu'),
            Dropout(0.4),
            Dense(4096, activation='relu'),
            Dropout(0.4),
            Dense(1000, activation='relu'),
            Dropout(0.4),
            Dense(17, activation='softmax')
        ])
        return model
    print("model was not LeNet or AlexNet")


def build_param(optType):
    if optType == 'Adam':
        param_dict = {
            # to assign to the learning rate
            'lr': random.random(),
            'lr_std': random.random(),
            'decay_steps': random.uniform(1, 100), # mutation will be + or - 1
            'decay': random.random(),
            'decay_std': random.random(),
            'staircase': random.choice([True, False]),

            'b1': random.random(),
            'b1_std': random.random(),
            'b2': random.random(),
            'b2_std': random.random(),
            'epsilon': random.random(),
            'epsilon_std': random.random(),
        }
        return param_dict
    elif optType == 'SGD':
        param_dict = {
            # to assign to the learning rate
            'lr': random.random(),
            'lr_std': random.random(),
            'decay_steps': random.uniform(1, 100), # mutation will be + or - 1
            'decay': random.random(),
            'decay_std': random.random(),
            'staircase': random.choice([True, False]),

            'momentum': random.random(),
            'momentum_std': random.random(),
            'nesterov': random.choice([True, False])
        }
        return param_dict
    print("optType was not Adam or SGD")
