"""Iterate over every combination of hyperparameters."""
import logging
from eeea.genetic.network import Network
from tqdm import tqdm
from itertools import product

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='brute-log.txt'
)

def train_networks(networks, dataset):
    """Train each network.
    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train(dataset)
        network.print_network()
        pbar.update(1)
    pbar.close()

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks.
    print_networks(networks[:5])

def print_networks(networks):
    """Print a list of networks.
    Args:
        networks (list): The population of networks
    """
    logging.info('-'*80)
    for network in networks:
        network.print_network()

def generate_network_list(nn_param_choices):
    """Generate a list of all possible networks.
    Args:
        nn_param_choices (dict): The parameter choices
    Returns:
        networks (list): A list of network objects
    """
    networks = []

    keys, possible_values = zip(*nn_param_choices.items())
    res = [dict(zip(keys, vals)) for vals in product(*possible_values)]

    for network in res:
        network_obj = Network()
        network_obj.create_set(network)
        networks.append(network_obj)

    return networks

def main():
    """Brute force test every network."""
    primitives = [
        'max_pool_3x3', 'avg_pool_3x3', 
        'skip_connect', 'sep_conv_3x3', 
        'sep_conv_5x5', 'dil_conv_3x3', 
        'inv_res_3x3' , 'inv_res_5x5' ,
        'conv_7x1_1x7', 'conv_1x1_3x3']
        
    number = [0,1,2]
    
    generations = 10
    population = 25 # Number of networks in each generation.
    dataset = 'cifar10'

    nn_param_choices = {
            'search': ['normal'], # 'normal','ee', 'ultra'
            'search_type': ['normal'], #' normal','progressive'
            'p_layers': [2],
            'layers': [8],
            'init_channels': [16],     
            'auxiliary': [True],
            'activation': ['relu'],
            'optimizer': ['sgd'], 
            'dropout_rate': [0.3],
            'autoaugment': [False],
            'th_param' : [2.9],
            'cutout' : [True],
            'parallel' : [False],
            'nc1' : primitives,
            'no1' : number[0:1],
            'nc2' : primitives,
            'no2' : number[1:2],
            'nc3' : primitives,
            'no3' : number,
            'nc4' : primitives,
            'no4' : number,
            'nc5' : primitives,
            'no5' : number,
            'nc6' : primitives,
            'no6' : number,
            'nc7' : primitives,
            'no7' : number,
            'nc8' : primitives,
            'no8' : number,
            'reduce_concat' : [range(2, 6), range(3, 6)],
            'normal_concat' : [range(2, 6), range(3, 6)],
            'rc1' : primitives,
            'ro1' : number[0:1],
            'rc2' : primitives,
            'ro2' : number[1:2],
            'rc3' : primitives,
            'ro3' : number,
            'rc4' : primitives,
            'ro4' : number,
            'rc5' : primitives,
            'ro5' : number,
            'rc6' : primitives,
            'ro6' : number,
            'rc7' : primitives,
            'ro7' : number,
            'rc8' : primitives,
            'ro8' : number,
        }

    logging.info("***Brute forcing networks***")

    networks = generate_network_list(nn_param_choices)

    train_networks(networks, dataset)

if __name__ == '__main__':
    main()