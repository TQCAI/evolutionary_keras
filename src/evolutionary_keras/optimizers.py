"""
    This module contains different Evolutionary Optimizers
"""

from abc import abstractmethod
from copy import deepcopy

import numpy as np
from keras.optimizers import Optimizer
from keras.utils.layer_utils import count_params
from numpy import empty, floor, fmax, identity, log, sqrt, transpose, zeros
from numpy.random import rand, randint, randn

from evolutionary_keras.utilities import (compatibility_numpy,
                                          get_number_nodes, parse_eval)


class EvolutionaryStrategies(Optimizer):
    """ Parent class for all Evolutionary Strategies
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        self.shape = None
        self.non_training_weights = []

    @abstractmethod
    def get_shape(self):
        """ Gets the shape of the weights to train """
        shape = None
        return shape

    def on_compile(self, model):
        """ Function to be called by the model during compile time.
        Register the model `model` with the optimizer.
        """
        # Here we can perform some checks as well
        self.model = model
        self.shape = self.get_shape()

    @abstractmethod
    def run_step(self, x, y):
        """ Performs a step of the optimizer.
        Returns the current score of the best mutant
        and its new weights """
        score = 0
        selected_parent = None
        return score, selected_parent

    def get_updates(self, loss, params):
        """ Capture Keras get_updates method """
        pass


class GA(EvolutionaryStrategies):
    pass


class NGA(EvolutionaryStrategies):
    """
    The Nodal Genetic Algorithm (NGA) is similar to the regular GA, but this time a number
    of nodes (defined by the mutation_rate variable) are selected at random and
    only the weights and biases corresponding to the selected nodes are mutated by
    adding normally distributed values with normal distrubtion given by sigma.

    Parameters
    ----------
        `sigma_init`: int
            Allows adjusting the original sigma
        `population_size`: int
            Number of mutants to be generated per iteration
        `mutation_rate`: float
            Mutation rate
    """

    # In case the user wants to adjust sigma_init
    # population_size or mutation_rate parameters the NGA method has to be initiated
    def __init__(self, sigma_init=15, population_size=80, mutation_rate=0.05, *args, **kwargs):
        self.sigma_init = sigma_init
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.sigma = sigma_init
        self.n_nodes = 0
        self.n_generations = 1

        super(NGA, self).__init__(*args, **kwargs)

    # Only works if all non_trainable_weights come after all trainable_weights
    # perhaps part of the functionality (getting shape) can be moved to ES
    def get_shape(self):
        """ Study the model to get the shapes of all trainable weight as well
        as the number of nodes. It also saves a reference to the non-trainable weights
        in the system.

        Returns
        -------
            `weight_shapes`: a list of the shapes of all trainable weights
        """
        # Initialize number of nodes
        self.n_nodes = 0
        # Get trainable weight from the model and their shapes
        trainable_weights = self.model.trainable_weights
        weight_shapes = [weight.shape.as_list() for weight in trainable_weights]
        # TODO: eventually we should save here a reference to the layer and their
        # corresponding weights, since the nodes are the output of the layer
        # and the weights the corresponding to that layer
        for layer in self.model.layers:
            self.n_nodes += get_number_nodes(layer)
        # TODO related to previous TODO: non trianable weights should not be important
        self.non_training_weights = self.model.non_trainable_weights
        return weight_shapes

    def create_mutants(self, change_both_wb=True):
        """
        Takes the current state of the network as the starting mutant and creates a new generation
        by performing random nodal mutations.
        By default, from a layer dense layer, only weights or biases will be mutated.
        In order to mutate both set `change_both_wb` to True
        """
        # TODO here we should get only trainable weights
        parent_mutant = self.model.get_weights()
        # Find out how many nodes are we mutating
        nodes_to_mutate = int(self.n_nodes * self.mutation_rate)

        mutants = [parent_mutant]
        for _ in range(self.population_size):
            mutant = deepcopy(parent_mutant)
            mutated_nodes = []
            # Select random nodes to mutate for this mutant
            # TODO seed numpy random at initialization time
            # Note that we might mutate the same node several times for the same mutant
            for _ in range(nodes_to_mutate):
                mutated_nodes.append(randint(self.n_nodes))

            for i in mutated_nodes:
                # Find the nodes in their respective layers
                count_nodes = 0
                layer = 1
                # TODO HERE WEIGHT-BIAS-WEIGHT-BIAS IS ALSO ASSUMED BY THE +=2
                # Once again, this is related to the previous TODO
                while count_nodes <= i:
                    count_nodes += self.shape[layer][0]
                    layer += 2
                layer -= 2
                count_nodes -= self.shape[layer][0]
                node_in_layer = i - count_nodes

                # Mutate weights and biases by adding values from random distributions
                sigma_eff = self.sigma * (self.n_generations ** (-rand()))
                if change_both_wb:
                    randn_mutation = sigma_eff * randn(self.shape[layer - 1][0])
                    mutant[layer - 1][:, node_in_layer] += randn_mutation
                    mutant[layer][node_in_layer] += sigma_eff * randn()
                else:
                    change_weight = randint(2, dtype="bool")
                    if change_weight:
                        randn_mutation = sigma_eff * randn(self.shape[layer - 1][0])
                        mutant[layer - 1][:, node_in_layer] += randn_mutation
                    else:
                        mutant[layer][node_in_layer] += sigma_eff * randn()
            mutants.append(mutant)
        return mutants

    # Evalutates all mutantants of a generationa and ouptus loss and the single best performing
    # mutant of the generation
    def evaluate_mutants(self, mutants, x=None, y=None, verbose=0):
        """ Evaluates all mutants of a generation and select the best one.

        Parameters
        ----------
            `mutants`: list of all mutants for this generation

        Returns
        -------
            `loss`: loss of the best performing mutant
            `best_mutant`: best performing mutant
        """
        best_loss = self.model.evaluate(x=x, y=y, verbose=verbose)
        best_loss_val = parse_eval(best_loss)
        best_mutant = mutants[0]
        new_mutant = False
        for mutant in mutants[1:]:
            # replace weights of the input model by weights generated
            # TODO related to the other todos, eventually this will have to be done
            # in a per-layer basis
            self.model.set_weights(mutant)
            loss = self.model.evaluate(x=x, y=y, verbose=False)
            loss_val = parse_eval(loss)

            if loss_val < best_loss_val:
                best_loss_val = loss_val
                best_loss = loss
                best_mutant = mutant
                new_mutant = True

        # if none of the mutants have performed better on the training data than the original mutant
        # reduce sigma
        if not new_mutant:
            self.n_generations += 1
            self.sigma = self.sigma_init / self.n_generations
        return best_loss, best_mutant

    # --------------------- only the functions below are called in EvolModel ---------------------
    def run_step(self, x, y):
        """ Wrapper to run one single step of the optimizer"""
        # Initialize training paramters
        mutants = self.create_mutants()
        score, selected_parent = self.evaluate_mutants(mutants, x=x, y=y)

        return score, selected_parent


class CMA(EvolutionaryStrategies):
    """
    From http://cma.gforge.inria.fr/:
    "The CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is an evolutionary algorithm for
    difficult non-linear non-convex black-box optimisation problems in continuous domain."
    The work-horse of this class is the cma package developed and maintained by Nikolaus Hansen
    (see https://pypi.org/project/cma/), this class allows for convenient implementation within
    the keras environment.

    Parameters
    ----------
        `sigma`: int
            Allows adjusting the initial sigma
        `population_size`: int
            Number of mutants to be generated per iteration
        `target_value`: float
            Stops the minimizer if the target loss is achieved
        `max_evaluations`: int
            Maximimum total number of mutants tested during optimization
    """

    def __init__(self, sigma=0.1, population_size=None, verbosity=1, *args, **kwargs):
        """
        As one might have noticed, 'CMA' does not allow the user to set a number of epochs, as this
        is dealth with by 'cma'. The default 'epochs' in EvolModel is one, meaning 'run step' is
        only called once during training.
        """
        self.sigma = sigma
        self.shape = None
        self.length_flat_layer = None
        self.trainable_weights_names = None
        self.population_size = population_size

        super(CMA, self).__init__(*args, **kwargs)

    def on_compile(self, model):
        """ Function to be called by the model during compile time. Register the model `model` with
        the optimizer.
        """
        self.model = model
        self.shape = self.get_shape()
        self.n = count_params(self.model.trainable_weights)

        self.counteval = 0
        if self.population_size is None:
            self.Lambda = int(4 + floor(3 * log(self.n)))
        else:
            self.Lambda = self.population_size
        print(f"The population size is {self.Lambda}")
        self.mu = int(self.Lambda / 2)
        self.wghts = log((self.Lambda + 1) / 2) - log([i + 1 for i in range(self.Lambda)])
        self.mueff = np.sum(self.wghts[: self.mu]) ** 2 / np.sum(self.wghts[: self.mu] ** 2)
        self.mueff_minus = np.sum(self.wghts[self.mu :]) ** 2 / np.sum(self.wghts[self.mu :] ** 2)

        alpha_cov = 2
        self.csigma = (self.mueff + 2) / (self.n + self.mueff + 5)
        self.dsigma = 1 + 2 * fmax(0, sqrt((self.mueff - 1) / (self.n + 1)) - 1) + self.csigma
        self.cc = (4 + self.mueff / self.n) / (self.n + 4 + 2 * self.mueff / self.n)
        self.c1 = alpha_cov / ((self.n + 1.3) ** 2 + self.mueff)
        cmupr = (
            alpha_cov
            * (self.mueff - 2 + 1 / self.mueff)
            / ((self.n + 2) ** 2 + alpha_cov * self.mueff / 2)
        )

        self.cmu = np.fmin(1 - self.c1, cmupr)
        self.alpha_mu_minus = 1 + self.c1 / self.cmu
        self.alpha_mueff_minus = 1 + (2 * self.mueff_minus) / (self.mueff + 2)
        self.alpha_posdef_minus = (1 - self.c1 - self.cmu) / (self.n * self.cmu)
        self.alpha_min = np.fmin(
            self.alpha_mu_minus, np.fmin(self.alpha_mueff_minus, self.alpha_posdef_minus)
        )

        self.eigenInterval = self.Lambda / ((self.c1 + self.cmu) * self.n * 10)

        self.wghts[: self.mu] /= np.sum(self.wghts[: self.mu])
        self.wghts[self.mu :] *= self.alpha_min / np.fabs(np.sum(self.wghts[self.mu :]))

        self.pc = zeros(self.n)
        self.ps = zeros(self.n)
        self.B = identity(self.n)
        self.D = identity(self.n)
        self.C = self.B @ self.D @ self.D @ self.B.T
        self.eigeneval = 0
        self.expN = sqrt(self.n) * (1 - 1 / (4 * self.n) + 1 / (21 * self.n ** 2))

    def get_shape(self):
        # we do all this to keep track of the position of the trainable weights
        self.trainable_weights_names = [weights.name for weights in self.model.trainable_weights]

        if self.trainable_weights_names == []:
            raise TypeError("The model does not have any trainable weights!")

        self.shape = [weight.shape.as_list() for weight in self.model.trainable_weights]
        return self.shape

    def weights_per_layer(self):
        """
        'weights_per_layer' creates 'self.lengt_flat_layer' which is a list conatining the numer of
        weights in each layer of the network.
        """

        # The first values of 'self.length_flat_layer' is set to 0 which is helpful in determining
        # the range of weights in the function 'undo_flatten'.
        self.length_flat_layer = [
            len(np.reshape(weight.numpy(), [-1])) for weight in self.model.trainable_weights
        ]
        self.length_flat_layer.insert(0, 0)

    def flatten(self):
        """
        'flatten' returns a 1 dimensional list of all weights in the keras model.
        """
        # The first values of 'self.length_flat_layer' is set to 0 which is helpful in determining
        # the range of weights in the function 'undo_flatten'.
        flattened_weights = []
        self.length_flat_layer = []
        self.length_flat_layer.append(0)
        for weight in self.model.trainable_weights:
            a = np.reshape(compatibility_numpy(weight), [-1])
            flattened_weights.append(a)
            self.length_flat_layer.append(len(a))

        flattened_weights = np.concatenate(flattened_weights)

        return flattened_weights

    def undo_flatten(self, flattened_weights):
        """
        'undo_flatten' does the inverse of 'flatten': it takes a 1 dimensional input and returns a
        weight structure that can be loaded into the model.
        """

        new_weights = []
        for i, layer_shape in enumerate(self.shape):
            flat_layer = flattened_weights[
                self.length_flat_layer[i] : self.length_flat_layer[i]
                + self.length_flat_layer[i + 1]
            ]
            new_weights.append(np.reshape(flat_layer, layer_shape))

        ordered_names = [weight.name for layer in self.model.layers for weight in layer.weights]

        new_parent = deepcopy(self.model.get_weights())
        for i, weight in enumerate(self.trainable_weights_names):
            location_weight = ordered_names.index(weight)
            new_parent[location_weight] = new_weights[i]

        return new_parent

    def minimizethis(self, weights, x, y):
        weights_model = self.undo_flatten(weights)
        self.model.set_weights(weights_model)
        loss = parse_eval(self.model.evaluate(x=x, y=y, verbose=0))
        return loss

    def run_step(self, x, y):
        """ Wrapper to the optimizer"""

        if self.counteval == 0:
            loss = parse_eval(self.model.evaluate(x=x, y=y, verbose=0))
            print(f"The initial loss is {loss}")

        self.counteval += 1

        self.xmean = self.flatten()
        arfitness = empty(self.Lambda)
        arz = empty((self.Lambda, self.n))
        arx = empty((self.Lambda, self.n))
        for i in range(self.Lambda):
            arz[i] = self.sigma * randn(self.n)
            arx[i] = self.xmean + self.sigma * self.B @ self.D @ arz[i]
            arfitness[i] = self.minimizethis(weights=arx[i], x=x, y=y)

        arindex = np.argsort(arfitness)
        self.xmean = arx.T @ self.wghts
        zmean = arz.T @ self.wghts

        self.ps = (1 - self.csigma) * self.ps + (
            sqrt(self.csigma * (2 - self.csigma) * self.mueff)
        ) * self.B @ zmean
        hl = np.linalg.norm(self.ps) / sqrt(1 - (1 - self.csigma)) ** (
            2 * self.counteval / self.Lambda
        )
        hr = (1.4 + 2 / (self.n + 1)) * self.expN
        if hl < hr:
            hsig = 1
        else:
            hsig = 0
        self.pc = (1 - self.cc) * self.pc + hsig * sqrt(
            self.cc * (2 - self.cc) * self.mueff
        ) * self.B @ self.D * zmean

        self.C = (
            (1 - self.c1 - self.cmu) * self.C
            + self.c1 * (self.pc @ self.pc.T + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
            + self.cmu
            * (self.B @ self.D @ arz.T)
            @ np.diag(self.wghts)
            @ (self.B @ self.D @ arz.T).T
        )

        self.sigma = self.sigma * np.exp(
            (self.csigma / self.dsigma) * (np.linalg.norm(self.ps) / self.expN - 1)
        )

        if self.counteval - self.eigeneval > self.eigenInterval:
            self.eigeneval = self.counteval
            self.C = np.triu(self.C) + np.transpose(np.triu(self.C, 1))
            self.D, self.B = np.linalg.eig(self.C)
            self.D = np.diag(sqrt(self.D))

        if (
            arfitness[arindex[0]]
            > (1 - 1 / 10000) * arfitness[arindex[int(np.ceil(0.7 * self.Lambda))]]
        ):
            self.sigma = self.sigma * np.exp(0.2 + self.csigma / self.dsigma)
            print("warning: flat fitness, consider reformulating the objective")

        # Transform 'xopt' to the models' weight shape.
        xopt = arx[arindex[0]]
        selected_parent = self.undo_flatten(xopt)

        # Determine the ultimatly selected mutants' performance on the training data.
        self.model.set_weights(selected_parent)
        score = arfitness[arindex[0]]
        print(f"score: {score}, \t sigma: {self.sigma}, \t epoch: {self.counteval}")
        return score, selected_parent


class BFGS(EvolutionaryStrategies):
    pass


class CeresSolver(EvolutionaryStrategies):
    pass


# wpr = []
# for i in range(Lambda):
#     wpr.append(log((Lambda + 1) / 2) - log(i + 1))
# mu = int(mu)

# psumwgt = 0
# nsumwgt = 0
# psumwgtsqr = 0
# nsumwgtsqr = 0
# for i in range(Lambda):
#     if i < mu:
#         psumwgt += wpr[i]
#         psumwgtsqr += wpr[i] ** 2
#     else:
#         nsumwgt += wpr[i]
#         nsumwgtsqr += wpr[i] ** 2

# mu_eff = psumwgt * psumwgt / psumwgtsqr
# mu_eff_minus = nsumwgt ** 2 / nsumwgtsqr

# alpha_cov = 2
# cmupr = alpha_cov * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + alpha_cov * mu_eff / 2)

# csigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
# dsigma = 1.0 + 2.0 * fmax(0, (sqrt((mu_eff - 1.0) / (n + 1.0))) - 1.0) + csigma
# cc = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
# c1 = alpha_cov / (pow(n + 1.3, 2.0) + mu_eff)
# cmu = min(1.0 - c1, cmupr)

# sumwgtpos = 0
# sumwgtneg = 0
# for i in range(Lambda):
#     if wpr[i] > 0:
#         sumwgtpos += wpr[i]
#     else:
#         sumwgtneg += fabs(wpr[i])

# alpha_mu_minus = 1.0 + c1 / cmu
# alpha_mueff_minus = 1.0 + (2 * mu_eff_minus) / (mu_eff + 2.0)
# alpha_posdef_minus = (1.0 - c1 - cmu) / (n * cmu)
# alpha_min = fmin(alpha_mu_minus, fmin(alpha_mueff_minus, alpha_posdef_minus))

# eigenInterval = (Lambda / (c1 + cmu) / n) / 10.0

# # ********************************** Normalising weights  **********************************
# wgts = zeros(Lambda)
# for i in range(Lambda):
#     if wpr[i] > 0:
#         wgts[i] = wpr[i] * 1 / sumwgtpos
#     else:
#         wgts[i] = wpr[i] * alpha_min / sumwgtneg

# sumtestpos = sum(wgts[:mu])
# sumtestneg = sum(wgts[mu:])
