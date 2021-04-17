from collections import namedtuple
# from eeea.genetic.optimizer import Optimizer
from eeea.firefly.optimizer import FAOptimizer
import logging
from tqdm import tqdm
import csv
import numpy as np
import os
import argparse
import eeea.utils.utils as utils
import glob
import sys
import math
import time
import hashlib
from tinydb import TinyDB, Query
import random
import eeea.utils.nasnet_setup as nasnet_setup
import json
import eeea.genetic.nsga2 as nsga2
import copy

result_acc_csv = []
result_params_csv = []
result_hash_csv = []
result_flops_csv = []
result_front_csv = []

parser = argparse.ArgumentParser("search")

# GA Config
parser.add_argument('--generations', type=int, default=30, help='generations')
parser.add_argument('--population', type=int, default=40, help='population')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100','imagenet'])
parser.add_argument('--search', type=str, default='ee', choices=['normal','ee', 'ultra'])
parser.add_argument('--search_type', type=str, default='normal', choices=['normal','progressive'])
parser.add_argument('--p_layers', type=int, default=2, help='progressive incresing every 2 layers')
parser.add_argument('--n_blocks', type=int, default=5, help='number of blocks in a cell')
parser.add_argument('--th_param', type=float, default=5.0, help='early exit')
parser.add_argument('--selection', type=str, default='tournament', choices=['random','tournament'])
parser.add_argument('--objective', type=str, default='multi', choices=['single','multi'])

parser.add_argument('--crossover_prob', type=float, default=0.1, help='crossover probability ')
parser.add_argument('--mutate_prob', type=float, default=0.1, help='mutate probability ')
parser.add_argument('--reject_prob', type=float, default=0.0, help='reject probability ')

parser.add_argument('--increment', type=int, default=6, help='filter increment')
parser.add_argument('--pyramid', action='store_true', default=True, help='pyramid')
parser.add_argument('--SE', action='store_true', default=True, help='enable SE')
parser.add_argument('--init_channels', type=int, default=36, help='init channels')
parser.add_argument('--init_channels_train', type=int, default=16, help='init channels')
parser.add_argument('--layers', type=int, default=8, help='layers')
parser.add_argument('--epochs', type=int, default=1, help='epochs to train')

# TinyDB
parser.add_argument('--load_db', action='store_true', default=True, help='load tinydb for fast search')

# God mode
parser.add_argument('--god_mode', action='store_true', default=True, help='god mode')

# Train Config

parser.add_argument('--data', type=str, default='datasets', help='location of the data corpus')
parser.add_argument('--tmp_data_dir', type=str, default='/home/mllab/proj/ILSVRC2015/Data/CLS-LOC', help='temp data dir')

parser.add_argument('--workers', type=int, default=8, help='number of workers to load dataset')
parser.add_argument('--batch_size_train', type=int, default=128, help='batch size')
parser.add_argument('--batch_size_val', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=True, help='enable cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--patience', type=int, default=5, help='patience of early stopping')

parser.add_argument('--auxiliary', action='store_true', default=True, help='enable auxiliary')

parser.add_argument('--activation', type=str, default='relu', choices=['relu'])
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd'])
parser.add_argument('--dropout_rate', type=float, default=0.3, help='dropout rate')
parser.add_argument('--autoaugment', action='store_true', default=False, help='enable Autoaugment')
parser.add_argument('--parallel', action='store_true', default=False, help='enable multi gpu parallel')
parser.add_argument('--mode', type=str, default='FP32', choices=['FP32', 'FP16'])
parser.add_argument('--seed', type=int, default=2020, help='seed random')
parser.add_argument('--save', type=str, default='Search', help='Search name')
parser.add_argument('--cutmix', action='store_true', default=False, help='use se')


args = parser.parse_args()

args.save = 'outputs/{}-{}-{}-{}-{}-{}'.format(args.save, args.objective, args.search, args.th_param, args.epochs, time.strftime("%Y%m%d-%H%M%S"))

utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

db_file = 'db/db_' + str(args.epochs) + ".json"
db = TinyDB(db_file)

best_genotypes = None

def train_networks(networks, generation):
    """Train each network.
    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.update_hash()

        if args.load_db == True:
            db_network = Query()
            match_network = db.search(db_network.hash == network.hash)
        
            if len(match_network) > 0:
                network.load_infomation(match_network[0]['accuracy'], match_network[0]['params'], match_network[0]['flops'])
                # Show Log
                print(match_network[0]['network'])
                network_mixed = nasnet_setup.decode_cell(match_network[0]['network'])
                logging.info(network_mixed)
                logging.info(match_network[0]['hash'])
                logging.info("param size = %fMB", match_network[0]['params'])
                logging.info('valid_acc %f', match_network[0]['accuracy'])

        if network.train_able:
            network.train(args, generation, db)
            
        pbar.update(1)
    pbar.close()

def train_one_networks(network, generation):
    network.update_hash()

    if args.load_db == True:
        db_network = Query()
        match_network = db.search(db_network.hash == network.hash)
    
        if len(match_network) > 0:
            network.load_infomation(match_network[0]['accuracy'], match_network[0]['params'], match_network[0]['flops'])
            # Show Log
            network_mixed = nasnet_setup.decode_cell(match_network[0]['network'])
            logging.info(network_mixed)
            logging.info(match_network[0]['hash'])
            logging.info("param size = %fMB", match_network[0]['params'])
            logging.info('valid_acc %f', match_network[0]['accuracy'])

    if network.train_able:
        network.train(args, generation, db)
            

def run_firefly(args, nn_param_choices, primitives):
    beta_init= 1
    beta_min=0.2
    gamma = 0.97
    
    max_bound = len(primitives)
    min_bound = 0
    alpha=0.2

    logging.info('init pop')

    optimizer = FAOptimizer(nn_param_choices, args)
    networks = optimizer.create_population()

    for generation in range(args.generations):
        logging.info('-'*80)
        logging.info("***Doing generation %d of %d***" %
                     (generation + 1, args.generations))
        logging.info('-'*80)

        train_networks(networks, generation)

        graded = [(fitness(args,network), network) for network in networks]
        population  = nsga2.RankAndCrowdingSurvival(graded , len(graded))

        alpha = _modify_alpha(args.generations, alpha)
        
        tmp_population = copy.deepcopy(population)
        for i in range(len(population)):
            for j in range(len(tmp_population)):
                if fitness(args, population[i]) < fitness(args, tmp_population[j]):
                    #print(population[j].hash)
                    #print(tmp_population[j].hash)
                    print(i,j)
                    pop_a = change_genome_to_index(population[i].network, primitives)
                    pop_b = change_genome_to_index(tmp_population[j].network, primitives)
                    problem_dim = len(pop_a)

                    r = math.sqrt(np.sum((pop_a - pop_b) ** 2))
                    beta = (beta_init - beta_min) * math.exp(-gamma * r ** 2) + beta_min
                    tmp = alpha * (np.random.random_sample((1, problem_dim))[0] - 0.5) * (max_bound - min_bound)
                    #tmp = alpha * (random_sample(problem_dim) - 0.5) * (max_bound - min_bound)
                    pop_tmp = check_position(pop_a * (1 - beta) + pop_b * beta + tmp, min_bound, max_bound)
                    pop_tmp = np.floor(pop_tmp).astype(int)

                    population[j].network = change_index_to_genome(pop_tmp, primitives)
                    population[j].update_hash()
                    population[j].train_able = True

                    print(population[j].network)

                    train_one_networks(population[j], generation)
                    #train_networks(population, generation)
        
        networks = population

        allacc, allparams, allhash, allflops, allfront = get_fitness_generation(networks, generation+1)

        result_acc_csv.append(allacc)
        result_params_csv.append(allparams)
        result_hash_csv.append(allhash)
        result_flops_csv.append(allflops)
        result_front_csv.append(allfront)

        save_population_json(networks, generation+1)


def fitness(args, network):
    if args.objective == 'single':
        return network.accuracy
    elif args.objective == 'multi':
        return ( 100-network.accuracy, network.params, network.flops)

def is_duplicate(genome, population):
    for i in range(0,len(population)):
        if (genome.hash == population[i].hash):
            return True
    return False

def check_position(position, min_bound, max_bound):
    position[position > max_bound] = max_bound
    position[position < min_bound] = min_bound
    return position

def _modify_alpha(generations, alpha):
    delta = 1 - (10 ** (-4) / 0.9) ** (1 / generations)
    alpha = (1 - delta) * alpha
    return alpha

def random_sample(size):
  return np.array(random.sample(range(100000000), k=size)) / 100000000

def change_genome_to_index(network, primitives):
  """
  Network -> "dict" {'0': 'dil_conv_5x5', '1': 0, '2': 'max_pool_3x3', '3': 0, ....
  Result ->  "list" [6, 0, 0, 0, 9, 0, 8, 0, 6, 2, 8, ......
  """
  result =[]
  for i in network:
      if int(i) % 2 == 0:
          network[i] = primitives.index(network[i])
  for i in network:
      result.append(network[i])
  return np.array(result)

def change_index_to_genome(network, primitives):
  """
  Network -> "list" [6, 0, 0, 0, 9, 0, 8, 0, 6, 2, 8, ......
  Result ->  "dict" {'0': 'dil_conv_5x5', '1': 0, '2': 'max_pool_3x3', '3': 0, ....
  """
  output = {}
  for i in range(len(network)):
    if i % 2 == 0:
      output[str(i)] = primitives[network[i]]
    else:
      output[str(i)] = network[i]
  return output

def save_population_json(networks, generation):
    json_db = TinyDB(os.path.join(args.save, 'data.json'))
    for network in networks:
        save_model = {
            "hash" : network.hash,
            "network" : network.network,
            "params"  : round(network.params, 2),
            "accuracy" : round(network.accuracy, 2),    
            "flops" : int(network.flops),
            "generation" : generation,
            "front" : network.front,
            "father" : network.father,
            "mother" : network.mother,
        }
        json_db.insert(save_model)

def save_best_network(networks):
    if args.objective == 'single':
        best_network = sorted(networks, key=lambda x: x.accuracy, reverse=True)[0]
    elif args.objective == 'multi':
        best_network = networks[0]
    
    global best_genotypes
    best_genotypes = "G"+str(best_network.hash)
    
    # file_genotypes = open('genotypes.py', 'a')
    # file_genotypes.write("\nG"+str(best_network.hash) + " = " + str(nasnet_setup.decode_cell(best_network.network))+"\n")
    # file_genotypes.close()

    logging.info("Best Network")
    
    logging.info(nasnet_setup.decode_cell(best_network.network))
    logging.info(best_network.hash)
    logging.info('valid_acc %f', best_network.accuracy)
    logging.info('params %f', best_network.params)
    logging.info('flops %f', best_network.flops)

def get_fitness_generation(networks, generation):
    result_accuracy = [generation]
    result_params = [generation]
    result_hash = [generation]
    result_flops = [generation]
    result_front = [generation]

    if args.objective == 'single':
        sortedNetwork = sorted(networks, key=lambda x: x.accuracy, reverse=True)
    elif args.objective == 'multi':
        sortedNetwork = networks

    for network in sortedNetwork:
        result_accuracy.append(round(float(network.accuracy), 2))
        result_params.append(round(float(network.params), 2))
        result_hash.append(str(network.hash))
        result_flops.append(round(float(network.flops), 2))
        result_front.append(int(network.front))

    output_accuracy = [",".join(str(x) for x in result_accuracy)]
    output_params = [",".join(str(x) for x in result_params)]
    output_hash = [",".join(str(x) for x in result_hash)]
    output_flops = [",".join(str(x) for x in result_flops)]
    output_front = [",".join(str(x) for x in result_front)]
    return output_accuracy, output_params, output_hash, output_flops, output_front

def get_average_att_network(networks, select='accuracy'):
    """Get the average accuracy for a group of networks.
    Args:
        networks (list): List of networks
    Returns:
        float: The average accuracy of a population of networks.
    """
    total = 0
    for network in networks:
        if select == 'accuracy':
            total += network.accuracy
        elif select == 'params':
            total += network.params
        elif select == 'flops':
            total += network.flops

    return total / len(networks)

def print_networks(networks):
    """Print a list of networks.
    Args:
        networks (list): The population of networks
    """
    logging.info('-'*80)
    for network in networks:
        network.print_network()

def main():
    np.random.seed(args.seed)
    random.seed(args.seed)

    utils.save_seed(args.seed, 'db/seed_val.txt')

    assert args.crossover_prob * args.population > 2, "Please increase crossover prob or population"

    logging.info("args = %s", args)

    primitives = [
    'max_pool_3x3', 'avg_pool_3x3', 
    'skip_connect', 'sep_conv_3x3', 
    'sep_conv_5x5', 'dil_conv_3x3', 
    'dil_conv_5x5', 'conv_7x1_1x7',
    'inv_res_3x3', 'inv_res_5x5', 
    'inv_res_3x3_sh', 'inv_res_5x5_sh', 
    'std_gn_3x3', 'std_gn_5x5',
    'mbconv_k3_t1', 'mbconv_k5_t1',
    'blur_pool_3x3'
    ]

    nas_setup = nasnet_setup.setup_NAS(args.n_blocks,len(primitives))

    nn_param_choices = nasnet_setup.create_param_choices(primitives, nas_setup)

    logging.info("***Evolving %d generations with population %d***" %
                 (args.generations, args.population))
    
    run_firefly(args, nn_param_choices, primitives)

if __name__ == '__main__':
    start= time.time()
    main()
    end = time.time()
    logging.info("Time = %fSec", (end - start))

    np.savetxt(os.path.join(args.save, 'result_acc.csv'), np.array(result_acc_csv), delimiter=',', header="Accuracy,Network", comments="", fmt='%s')
    np.savetxt(os.path.join(args.save, 'result_params.csv'), np.array(result_params_csv), delimiter=',', header="Params,Network", comments="", fmt='%s')
    np.savetxt(os.path.join(args.save, 'result_hash.csv'), np.array(result_hash_csv), delimiter=',', header="Hash,Network", comments="", fmt='%s')
    np.savetxt(os.path.join(args.save, 'result_flops.csv'), np.array(result_flops_csv), delimiter=',', header="Flops,Network", comments="", fmt='%s')
    np.savetxt(os.path.join(args.save, 'result_front.csv'), np.array(result_front_csv), delimiter=',', header="Front,Network", comments="", fmt='%s')

    #import train_cifar
    #train_cifar.main(best_genotypes)
    
