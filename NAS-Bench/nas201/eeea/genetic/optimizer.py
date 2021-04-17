import random
from functools import reduce
from operator import add
from collections import namedtuple
from eeea.network.model import NetworkCIFAR as NetworkCIFAR
from eeea.network.model import PyramidNetworkCIFAR as PyramidNetwork
import eeea.utils.utils as utils
import eeea.genetic.nsga2 as nsga2
from eeea.genetic.network import Network
import eeea.network.genotypes as genotypes
import eeea.utils.nasnet_setup as nasnet_setup

seed = utils.load_seed('db/seed_val.txt')
random.seed(seed)

class Optimizer():
    """Class that implements genetic algorithm for MLP optimization."""

    def __init__(self, nn_param_choices, args, api):
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
        self.inversion_chance = args.inversion_prob
        self.mutate_chance = args.mutate_prob
        self.random_select = args.reject_prob
        self.retain = args.crossover_prob 
        self.nn_param_choices = nn_param_choices
        self.args = args
        self.populations = []
        self.api = api

    def is_duplicate(self, genome, population):
        """Add the genome to our population.
        """

        for i in range(0,len(population)):
            if (genome.hash == population[i].hash):
                return True
    
        return False

    def get_sorted_networks(self, networks):
        sortedNetwork = [(self.fitness(network), network) for network in networks]
        return nsga2.get_sorted_RankAndCrowdingSurvival(sortedNetwork, len(sortedNetwork))

    def params_threshold(self, networks, args):
        arch = nasnet_setup.decode(networks)
        index = self.api.query_index_by_arch(arch.tostr())
        results = self.api.query_by_index(index, 'cifar100')
        for seed, result in results.items():
            modelLarge = result.params
    
        return modelLarge >= args.th_param
    
    def get_params(self, networks, args):
        arch = nasnet_setup.decode(networks)
        index = self.api.query_index_by_arch(arch.tostr())
        results = self.api.query_by_index(index, 'cifar100')
        for seed, result in results.items():
            modelLarge = result.params
        return modelLarge

    def create_population(self):
        """Create a population of random networks.
        Args:
            count (int): Number of networks to generate, aka the
                size of the population
        Returns:
            (list): Population of network objects
        """
        pop = []
        for _ in range(0, self.args.population):
            # Create a random network.
            network = Network(self.nn_param_choices, self.args.seed)
            
            if self.args.search == 'ee' and self.args.th_param > 0 :
                network.create_random()
                print(network.hash, self.get_params(network.network, self.args))
                while(self.params_threshold(network.network, self.args)):
                    network.create_random()
                    print(network.hash, self.get_params(network.network, self.args), len(pop))
            else:
                network.create_random()

            if not self.is_duplicate(network, pop):
                network.update_hash()
                # Add the network to our population.
                pop.append(network)
            
        if self.args.god_mode:
            #adam_eve = ['AmoebaNet', 'EEEA_L']
            adam_eve = ['DARTS_V1', 'DARTS_V2', 'SNAS', 'PDARTS', 'PC_DARTS', 'CDARTS', 'CARS_I', 'CARS_H']
            pop = pop[:(len(pop)-len(adam_eve))]
            for model in adam_eve:
                genotype = eval("genotypes.%s" % model)
                network = Network(self.nn_param_choices, self.args.seed)
                network.network = nasnet_setup.decode_genotype(genotype)
                network.update_hash()
                pop.append(network)

        self.populations.extend(pop)
        return pop
    
    # @staticmethod
    def fitness(self, network):
        """Return the accuracy, which is our fitness function."""
        if self.args.objective == 'single':
            return network.accuracy
        elif self.args.objective == 'multi':
            return ( 100-network.accuracy, network.params, network.flops) #

    def grade(self, pop):
        """Find average fitness for a population.

        Args:
            pop (list): The population of networks

        Returns:
            (float): The average accuracy of the population

        """
        summed = reduce(add, (self.fitness(network) for network in pop))
        return summed / float((len(pop)))

    def breed(self, mother, father):
        """Make two children as parts of their parents.

        Args:
            mother (dict): Network parameters
            father (dict): Network parameters

        Returns:
            (list): Two network objects

        """
        children = []
        for _ in range(2):

            child = {}

            # Loop through the parameters and pick params for the kid.
            for param in self.nn_param_choices:
                child[param] = random.choice(
                    [mother.network[param], father.network[param]]
                )

            # Now create a network object.
            network = Network(self.nn_param_choices)
            network.create_set(child)

            # Randomly mutate some of the children.
            if self.mutate_chance > random.random():
                network = self.mutate(network)

            # Randomly inversion some of the children.
            if self.inversion_chance > 0:
                if self.inversion_chance > random.random():
                    network = self.inversion(network)

            network.father = father.hash
            network.mother = mother.hash

            network.update_hash()

            children.append(network)

        return children

    def mutate(self, network):
        """Randomly mutate one part of the network.

        Args:
            network (dict): The network parameters to mutate

        Returns:
            (Network): A randomly mutated network object

        """
        lists_keys = list(self.nn_param_choices.keys())
        length_keys = len(lists_keys)
        number_random = random.randrange(0, length_keys, 2)

        # Choose a random key.
        #mutation = random.choice(list(self.nn_param_choices.keys()))
        mutation = lists_keys[number_random]
        mutation_path = lists_keys[number_random+1]

        # Mutate one of the params.
        network.network[mutation] = random.choice(self.nn_param_choices[mutation])

        # Mutate one of the params.
        network.network[mutation_path] = random.choice(self.nn_param_choices[mutation_path])

        # Update Hash Network
        network.update_hash()

        # Update Trainable
        network.train_able = True

        return network

    def inversion(self, network):   
        lists_keys = list(self.nn_param_choices.keys())
        length_keys = len(lists_keys)
        number_random = random.randrange(0, length_keys, 2)

        # Choose a random key.
        if number_random == (length_keys-2):
            inversion_start = lists_keys[number_random-2]
            inversion_start_path = lists_keys[(number_random-2)+1]

            inversion_end = lists_keys[number_random]
            inversion_end_path = lists_keys[number_random+1]
        else:
            inversion_start = lists_keys[number_random]
            inversion_start_path = lists_keys[number_random+1]

            inversion_end = lists_keys[number_random+2]
            inversion_end_path = lists_keys[(number_random+2)+1]
        
        network.network[inversion_start] = network.network[inversion_end]
        network.network[inversion_end] = network.network[inversion_start]

        network.network[inversion_start_path] = network.network[inversion_end_path]
        network.network[inversion_end_path] = network.network[inversion_start_path]

        network.update_hash()

        network.train_able = True

        return network

    def tournament_selection(self, parents, k):
        rand_pop = random.sample(range(len(parents)), k=k*2)
        if k == 1:
            return parents[rand_pop[0]], parents[rand_pop[1]]
        else:
            male, female = None, None
            for i in range(0,k):
                pop = parents[rand_pop[i]]
                if male == None or nsga2.dominates(self.fitness(pop), self.fitness(male)):
                    male = pop
        
            for i in range(k,k*2):
                pop = parents[rand_pop[i]]
                if female == None or nsga2.dominates(self.fitness(pop), self.fitness(female)):
                    female = pop

            return male, female

    def survival(self, pop):
        # Get scores for each network.
        graded = [(self.fitness(network), network) for network in pop]
        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded)*self.retain)

        if self.args.objective == 'single':
            # Sort on the scores.
            graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]
            
        elif self.args.objective == 'multi':
            graded = nsga2.RankAndCrowdingSurvival(graded, len(graded))
        
        parents = graded[:retain_length]
        # For those we aren't keeping, randomly keep some anyway.
        # for individual in graded[retain_length:]:
        #     if self.random_select > random.random():
        #         parents.append(individual)
        
        return parents

    def evolve(self, pop):
        """Evolve a population of networks.

        Args:
            pop (list): A list of network parameters

        Returns:
            (list): The evolved population of networks

        """

        parents = self.survival(pop)

        # Now find out how many spots we have left to fill.
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []

        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:
            
            # selection 
            if self.args.selection == 'random' :
                male, female = self.tournament_selection(parents, k=1)
            else:
                male, female = self.tournament_selection(parents, k=int(len(parents)/2))
            # Breed them.
            babies = self.breed(male, female)

            # Add the children one at a time.
            for baby in babies:
                # Don't grow larger than desired length.
                if len(children) < desired_length:
                    if (not self.is_duplicate(baby, children)) and (not self.is_duplicate(baby, parents)) and (not self.is_duplicate(baby, self.populations)):
                        """
                        if self.args.search == 'ee' and self.args.th_param > 0 :
                            if self.params_threshold(baby.network, self.args):
                                break
                            else:
                                children.append(baby)
                        else:
                        """
                        children.append(baby)

        parents.extend(children)
        self.populations.extend(parents)
        return parents