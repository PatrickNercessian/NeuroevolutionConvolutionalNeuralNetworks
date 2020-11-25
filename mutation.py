import random
import math


def gaussian_mutate(indiv, indpb, param_bounds):
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
