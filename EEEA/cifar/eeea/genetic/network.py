"""Class that represents the network to be evolved."""
import random
import logging
import hashlib
from eeea.engine.search import train_and_score
from tinydb import TinyDB, Query
import eeea.utils.utils as utils
import numpy as np
import json
import eeea.utils.nasnet_setup as nasnet_setup

seed = utils.load_seed('db/seed_val.txt')
random.seed(seed)

class Network():
    """Represent a network and let us operate on it.

    Currently only works for an MLP.
    """

    def __init__(self, nn_param_choices=None, seed=0):
        """Initialize our network.

        Args:
            nn_param_choices (dict): Parameters for the network, includes:
                nb_neurons (list): [64, 128, 256]
                nb_layers (list): [1, 2, 3, 4]
                activation (list): ['relu', 'elu']
                optimizer (list): ['rmsprop', 'adam']
        """
        self.accuracy = 0.
        self.params = 0
        self.nn_param_choices = nn_param_choices
        self.network = {}  # (dic): represents MLP network parameters
        self.hash = ''
        self.train_able = True
        self.flops = 0
        self.front = False
        self.father = None
        self.mother = None
        self.epoch_time = 0
        
    def create_random(self):
        """Create a random network."""

        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])
        self.hash = hashlib.md5(str(self.network).encode("UTF-8")).hexdigest()

    def update_hash(self):
        self.hash = hashlib.md5(str(self.network).encode("UTF-8")).hexdigest()

    def create_set(self, network):
        """Set network properties.

        Args:
            network (dict): The network parameters

        """
        self.network = network

    def train(self, dataset, generation, db):
        """Train the network and record the accuracy.

        Args:
            dataset (str): Name of dataset to use.

        """
        #if self.accuracy == 0.:
        self.accuracy , self.params, self.flops, self.epoch_time = train_and_score(self.network, dataset, generation, self.hash)
        self.train_able = False

        save_model = {
            "hash" : self.hash,
            "network" : self.network,
            "params"  : round(self.params, 2),
            "accuracy" : round(self.accuracy, 2),    
            "flops" : round(self.flops, 4), #GFLOPS
            "epoch_time": self.epoch_time,
        }

        if self.accuracy > 0:
            db.insert(save_model)

    def print_network(self):
        """Print out a network."""
        logging.info(self.accuracy, self.params, self.network)
    
    def load_infomation(self, accuracy, params, flops, epoch_time):
        self.accuracy = accuracy
        self.params = params
        self.flops = flops
        self.train_able = False
        self.epoch_time = epoch_time