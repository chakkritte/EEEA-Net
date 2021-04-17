import os
import sys
import numpy as np
import time
import torch
import eeea.utils.utils as utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import eeea.network.genotypes as genotypes 
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from eeea.network.model import NetworkImageNet as Network
from eeea.network.model import PyramidNetworkImageNet as PyramidNetwork

from tensorboard_logger import configure, log_value
from eeea.utils.flops_counter import get_model_complexity_info
from eeea.data.folder2lmdb import ImageFolderLMDB

import eeea.data.datasets as data
import eeea.engine.inference as inference
import eeea.engine.trainer as trainer

parser = argparse.ArgumentParser("training imagenet")
parser.add_argument('--workers', type=int, default=16, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.5, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')

parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='Gb728e3edb4140725a222c6e2df7effeb', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--lr_scheduler', type=str, default='linear', help='lr scheduler, linear or cosine')
parser.add_argument('--tmp_data_dir', type=str, default='/home/mllab/proj/ILSVRC2015/Data/CLS-LOC', help='temp data dir')
parser.add_argument('--note', type=str, default='imagenet', help='note for this run')
parser.add_argument('--tensorboard', help='Log progress to TensorBoard', default=True, action='store_true')
parser.add_argument('--lmdb', action='store_true', help='Use lmdb to load data')

parser.add_argument('--increment', type=int, default=8, help='filter increment')
parser.add_argument('--pyramid', action='store_true', default=False, help='pyramid')
parser.add_argument('--se', action='store_true', default=False, help='use se')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')

parser.add_argument('--resume', type=str, default='eval-imagenet-20200417-060401/model_best.pth.tar', help='path of pretrained model')
parser.add_argument('--start_epoch', type=int, default=0, help='start')

args, unparsed = parser.parse_known_args()

args.save = '{}eval-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CLASSES = 1000

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss

def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)
    logging.info("unparsed_args = %s", unparsed)
    num_gpus = torch.cuda.device_count()

    if args.tensorboard:
        print(args.save)
        configure(args.save)

    genotype = eval("genotypes.%s" % args.arch)
    print('---------Genotype---------')
    logging.info(genotype)
    print('--------------------------') 

    if args.pyramid == False:
        model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype, args.drop_path_prob, "FP32")
    else:
        model = PyramidNetwork(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype, args.drop_path_prob, "FP32", args.se, args.increment)

    from torch.autograd import Variable
    x_image = Variable(torch.randn(1, 3, 224, 224))
    y = model(x_image)
    flops, params = get_model_complexity_info(model, (224, 224), as_strings=True, print_per_layer_stat=False)
    # print('Flops:  {}'.format(flops))
    # print('Params: ' + params)
    logging.info("Flops:  %s", flops)

    if num_gpus > 1:
        model = nn.DataParallel(model)
        model = model.cuda()
    else:
        model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
        )

        
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_acc_top1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    train_queue, valid_queue = data.get_imagenet_queue(args)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    best_acc_top1 = 0
    best_acc_top5 = 0

    if args.resume:
        if os.path.isfile(args.resume):
            lr = adjust_lr(optimizer, args.start_epoch-1)	
        else:
            lr = args.learning_rate

    print(args.start_epoch, lr)
    for epoch in range(args.start_epoch, args.epochs):	

        if epoch < 5 and args.batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (epoch + 1) / 5.0
            logging.info('Warming-up Epoch: %d, LR: %e', epoch, lr * (epoch + 1) / 5.0)
        if num_gpus > 1:
            model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        else:
            model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        
        # log to TensorBoard
        if args.tensorboard:
            log_value('learning_rate', get_learning_rate(optimizer)[0], epoch)
        
        logging.info('Epoch: %d lr %e', epoch, get_learning_rate(optimizer)[0])

        epoch_start = time.time()

        train_acc, train_obj = trainer.train_imagenet(args, train_queue, model, criterion, optimizer, logging_mode=True)

        # log to TensorBoard
        if args.tensorboard:
            log_value('train_loss', train_obj, epoch)
            log_value('train_acc', train_acc, epoch)

        logging.info('Train_acc: %f', train_acc)

        valid_acc_top1, valid_acc_top5, valid_obj = inference.infer_imagenet(args, valid_queue, model, criterion_smooth, logging_mode=True)

        # log to TensorBoard
        if args.tensorboard:
            log_value('val_loss', valid_obj, epoch)
            log_value('val_acc_top1', valid_acc_top1, epoch)
            log_value('val_acc_top5', valid_acc_top5, epoch)

        logging.info('Valid_acc_top1: %f', valid_acc_top1)
        logging.info('Valid_acc_top5: %f', valid_acc_top5)
        epoch_duration = time.time() - epoch_start
        logging.info('Epoch time: %ds.', epoch_duration)
        is_best = False
        if valid_acc_top5 > best_acc_top5:
            best_acc_top5 = valid_acc_top5
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True

        if args.lr_scheduler == 'cosine':	
            scheduler.step()	
            current_lr = scheduler.get_lr()[0]	
        elif args.lr_scheduler == 'linear':	
            current_lr = adjust_lr(optimizer, epoch)	
        else:	
            print('Wrong lr type, exit')	
            sys.exit(1)

        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc_top1': best_acc_top1,
            'optimizer' : optimizer.state_dict(),
            }, is_best, args.save)        
        
def adjust_lr(optimizer, epoch):
    # Smaller slope for the last 5 epochs because lr * 1/250 is relatively large
    if args.epochs -  epoch > 5:
        lr = args.learning_rate * (args.epochs - 5 - epoch) / (args.epochs - 5)
    else:
        lr = args.learning_rate * (args.epochs - epoch) / ((args.epochs - 5) * 5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr        

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
        lr +=[ param_group['lr'] ]
    return lr

if __name__ == '__main__':
    main()
