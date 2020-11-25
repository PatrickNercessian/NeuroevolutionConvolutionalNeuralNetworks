import random

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