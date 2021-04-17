import random
import numpy as np
import math
import eeea.utils.utils as utils
from eeea.genetic.network import Network
import eeea.genetic.nsga2 as nsga2
import copy

seed = utils.load_seed('db/seed_val.txt')
random.seed(seed)

class FAOptimizer():
    def __init__(self, nn_param_choices, args):
        """Create an optimizer.
        Args:
            nn_param_choices (dict): Possible network paremters
            retain (float): Percentage of population to retain after
                each generation
            random_select (float): Probability of a rejected network
                remaining in the population
            mutate_chance (float): Probability a network will be
                randomly mutated
        """
        self.nn_param_choices = nn_param_choices
        self.args = args

    def is_duplicate(self, genome, population):
        """Add the genome to our population.
        """

        for i in range(0,len(population)):
            if (genome.hash == population[i].hash):
                return True
    
        return False

    def fitness(self, network):
        """Return the accuracy, which is our fitness function."""
        if self.args.objective == 'single':
            return network.accuracy
        elif self.args.objective == 'multi':
            return ( 100-network.accuracy, network.params, network.flops)

    def create_population(self):
        pop = []
        for _ in range(0, self.args.population):
        # Create a random network.
            network = Network(self.nn_param_choices, self.args.seed)
            network.create_random()

            if not self.is_duplicate(network, pop):
                network.update_hash()
                # Add the network to our population.
                pop.append(network)
        return pop