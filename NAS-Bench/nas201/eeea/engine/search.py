import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from collections import namedtuple
from eeea.network.model import NetworkCIFAR as Network
from eeea.network.model import PyramidNetworkCIFAR as PyramidNetwork
from eeea.network.model import NetworkImageNet as NetworkImageNet
from eeea.network.model import PyramidNetworkImageNet as PyramidNetworkImageNet

import eeea.data.datasets as data

import eeea.engine.inference as inference
import eeea.engine.trainer as trainer

# import EarlyStopping
from eeea.utils.earlystopping import EarlyStopping
from eeea.utils.flops_counter import get_flops_params
import eeea.utils.nasnet_setup as nasnet_setup
import eeea.utils.utils as utils

patience = 5

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

def train_and_score(network, args, generation, network_hash):
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    #cudnn.deterministic = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)

    torch.cuda.manual_seed_all(args.seed)
    #torch.backends.cudnn.deterministic = True

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    if args.dataset == 'cifar10':
        Layers_Large = 20
    if args.dataset == 'imagenet':
        Layers_Large = 14

    network_mixed = nasnet_setup.decode_cell(network)
    logging.info(network_mixed)
    logging.info(network_hash)

    number_layers = args.layers
    if args.search_type == 'progressive':
        number_layers = utils.progressive_layer(args)[generation]
        # number_layers = args.p_layers * (generation+1)

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        Layers_Large = 20
        if args.pyramid == False:
            model = Network(args.init_channels_train, args.classes, number_layers, args.auxiliary, network_mixed, args.dropout_rate, args.mode, args.SE)
            modelLarge = Network(args.init_channels, args.classes, Layers_Large, args.auxiliary, network_mixed, args.dropout_rate, args.mode, args.SE)
        else:
            model = PyramidNetwork(args.init_channels_train, args.classes, number_layers, args.auxiliary, network_mixed, args.dropout_rate, args.mode, args.SE, args.increment)
            modelLarge = PyramidNetwork(args.init_channels, args.classes, Layers_Large, args.auxiliary, network_mixed, args.dropout_rate, args.mode, args.SE, args.increment)       

    if args.dataset == 'imagenet':
        Layers_Large = 14
        if args.pyramid == False:
            model = NetworkImageNet(args.init_channels_train, args.classes, number_layers, args.auxiliary, network_mixed, args.dropout_rate, args.mode)
            modelLarge = NetworkImageNet(args.init_channels, args.classes, Layers_Large, args.auxiliary, network_mixed, args.dropout_rate, args.mode)
        else:
            model = PyramidNetworkImageNet(args.init_channels_train, args.classes, number_layers, args.auxiliary, network_mixed, args.dropout_rate, args.mode, args.SE, args.increment)
            modelLarge = PyramidNetworkImageNet(args.init_channels, args.classes, Layers_Large, args.auxiliary, network_mixed, args.dropout_rate, args.mode, args.SE, args.increment)       

    logging.info("param size = %fMB", utils.count_parameters_in_MB(modelLarge))
    
    if args.parallel == True:
        model = nn.DataParallel(model).cuda()
    else:
        model = model.cuda()
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        nesterov=True,
        weight_decay=args.weight_decay
    )

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        train_queue, valid_queue = data.get_cifar_queue(args)

    if args.dataset == 'imagenet':
        train_queue, valid_queue = data.get_imagenet_search_queue(args)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    epoch_time = 0
    last_valid_acc = 0
    for epoch in range(args.epochs):
        if args.search == 'ee' and args.th_param > 0 or args.search == 'ultra':
            if utils.count_parameters_in_MB(modelLarge) >= args.th_param:
                print("Drop model")
                break

        b_start = time.time()

        #logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.drop_path_prob = args.dropout_rate * epoch / args.epochs
        
        if args.dataset == 'cifar10' or args.dataset == 'cifar100' :
            train_acc, train_obj = trainer.train_cifar(args, train_queue, model, criterion, optimizer, logging_mode=False)
            valid_acc, valid_obj = inference.infer_cifar(args, valid_queue, model, criterion, logging_mode=False)

        if args.dataset == 'imagenet':
            train_acc, train_obj = trainer.train_imagenet(args, train_queue, model, criterion, optimizer, logging_mode=False)
            valid_acc, valid_obj = inference.infer_imagenet(args, valid_queue, model, criterion, logging_mode=False)
        last_valid_acc = valid_acc

        print('Test (on val set): [{0}/{1}]\tLoss {2}\tTop 1-acc {3}'.format(epoch, args.epochs, valid_obj, valid_acc))

        #utils.save(model, os.path.join(SAVE, 'weights.pt'))
        scheduler.step()

        epoch_time = time.time() - b_start

        early_stopping(valid_obj, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    logging.info('valid_acc %f', last_valid_acc)

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        flops, params = get_flops_params(modelLarge, (32, 32))
    elif args.dataset == 'imagenet':
        flops, params = get_flops_params(modelLarge, (224, 224))

    logging.info("Flops = %f GFLOPs", flops/10.**9)
    logging.info("Epoch Time = %f sec", round(epoch_time, 2))
    
    return last_valid_acc, utils.count_parameters_in_MB(modelLarge), flops/10.**9, round(epoch_time, 2)