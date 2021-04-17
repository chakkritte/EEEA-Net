import os
import math
import time
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from tensorboard_logger import configure, log_value

import eeea.network.genotypes as genotypes
import eeea.data.datasets as data
import eeea.engine.inference as inference
import eeea.engine.trainer as trainer
import eeea.utils.nasnet_setup as nasnet_setup
import eeea.utils.utils as utils

from eeea.network.model import NetworkCIFAR as Network
from eeea.network.model import PyramidNetworkCIFAR as PyramidNetwork
from eeea.utils.flops_counter import get_model_complexity_info

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--workers', type=int, default=8, help='number of workers to load dataset')
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size_train', type=int, default=96, help='batch size')
parser.add_argument('--batch_size_val', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')

parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='EEEA_L', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')
parser.add_argument('--mode', type=str, default='FP32', choices=['FP32', 'FP16', 'amp'])
parser.add_argument('--tensorboard', help='Log progress to TensorBoard', default=True, action='store_true')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100','cinic10'])
parser.add_argument('--classes', type=int, default=10, help='classes')
parser.add_argument('--duplicates', type=int, default=1, help='duplicates')

parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'amsbound'])

parser.add_argument('--warmup', action='store_true', default=False, help='use warmup')

parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--increment', type=int, default=6, help='filter increment')
parser.add_argument('--pyramid', action='store_true', default=False, help='pyramid')

parser.add_argument('--autoaugment', action='store_true', default=False, help='use autoaugment')
parser.add_argument('--se', action='store_true', default=False, help='use se')
parser.add_argument('--cutmix', action='store_true', default=False, help='use cutmix')

args = parser.parse_args()

args = utils.get_classes(args)

if args.dataset == 'cifar10':
    args.save = 'outputs/eval-cifar10-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
if args.dataset == 'cifar100':
    args.save = 'outputs/eval-cifar100-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
if args.dataset == 'cinic10':
    args.save = 'outputs/eval-cinic10-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))

def init():
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

best_acc1 = 0

def main(arch):
    start= time.time()
    
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    
    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    init()
    args.arch = arch

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)

    global best_acc1

    if args.tensorboard:
        print(args.save)
        configure(args.save)

    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    genotype = eval("genotypes.%s" % args.arch)

    if args.pyramid == False:
        model = Network(args.init_channels, args.classes, args.layers, args.auxiliary, genotype, args.drop_path_prob, args.mode, args.se)
    else:
        model = PyramidNetwork(args.init_channels, args.classes, args.layers, args.auxiliary, genotype, args.drop_path_prob, args.mode, args.se, args.increment)

    flops, params = get_model_complexity_info(model, (32, 32), as_strings=True, print_per_layer_stat=False)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    logging.info("flops = %s", flops)

    if args.parallel == True:
        model = nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    train_queue, valid_queue = data.get_cifar_queue(args)

    if args.warmup:
        from warmup_scheduler import GradualWarmupScheduler
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-5, eta_min=0.0)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler_cosine)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=0.0)

    for epoch in range(args.epochs):
        logging.info('epoch %d lr %e', epoch, get_learning_rate(optimizer)[0])
        # log to TensorBoard
        if args.tensorboard:
            log_value('learning_rate', get_learning_rate(optimizer)[0], epoch)
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = trainer.train_cifar(args, train_queue, model, criterion, optimizer, logging_mode=True)
        # log to TensorBoard
        if args.tensorboard:
            log_value('train_loss', train_obj, epoch)
            log_value('train_acc', train_acc, epoch)
        logging.info('train_acc %f', train_acc)

        valid_acc, valid_obj = inference.infer_cifar(args, valid_queue, model, criterion, logging_mode=True)
        # log to TensorBoard
        if args.tensorboard:
            log_value('val_loss', valid_obj, epoch)
            log_value('val_acc', valid_acc, epoch)
        logging.info('valid_acc %f', valid_acc)
        scheduler.step()
        # remember best acc@1 and save checkpoint
        is_best = valid_acc > best_acc1
        best_acc1 = max(valid_acc, best_acc1)

        utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, args.save)

        utils.save(model, os.path.join(args.save, 'weights.pt'))

    logging.info('best_acc1 %f', best_acc1)
    end = time.time()
    logging.info("Time = %fSec", (end - start))


def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]
    return lr

if __name__ == '__main__':
  
  main(args.arch)