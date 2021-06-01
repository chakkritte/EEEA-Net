import logging
import eeea.utils.utils as utils
import torch.nn as nn
import torch
import time
import numpy as np
import random

def train_cifar(args, train_queue, model, criterion, optimizer, logging_mode=False):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    model.train()

    for step, (input, target) in enumerate(train_queue):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        
        optimizer.zero_grad()

        r = np.random.rand(1)
        cutmix_beta = 1.0
        if args.cutmix and cutmix_beta > 0 and r < 0.5:
            # generate mixed sample
            lam = np.random.beta(cutmix_beta, cutmix_beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            # compute output
            input_var = torch.autograd.Variable(input, requires_grad=True)
            target_a_var = torch.autograd.Variable(target_a)
            target_b_var = torch.autograd.Variable(target_b)

            logits, logits_aux = model(input_var)
            loss = criterion(logits, target_a_var) * lam + criterion(logits, target_b_var) * (1. - lam)
            if args.auxiliary:
                loss_aux = criterion(logits_aux, target_a_var) * lam + criterion(logits_aux, target_b_var) * (1. - lam)
                loss += args.auxiliary_weight*loss_aux
        else:
            logits, logits_aux = model(input)
            loss = criterion(logits, target)
            if args.auxiliary:
                loss_aux = criterion(logits_aux, target)
                loss += args.auxiliary_weight*loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        
        if logging_mode:
            if step % args.report_freq == 0:
                logging.info('train %03d %f %f %f mem %.2f MB', step, objs.avg, top1.avg, top5.avg, round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))

    return top1.avg, objs.avg


def train_imagenet(args, train_queue, model, criterion, optimizer, logging_mode=False):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    batch_time = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        b_start = time.time()
        optimizer.zero_grad()
        
        r = np.random.rand(1)
        cutmix_beta = 1.0
        if args.cutmix and cutmix_beta > 0 and r < 0.5:
            # generate mixed sample
            lam = np.random.beta(cutmix_beta, cutmix_beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            # compute output
            input_var = torch.autograd.Variable(input, requires_grad=True)
            target_a_var = torch.autograd.Variable(target_a)
            target_b_var = torch.autograd.Variable(target_b)

            logits, logits_aux = model(input_var)
            loss = criterion(logits, target_a_var) * lam + criterion(logits, target_b_var) * (1. - lam)
            if args.auxiliary:
                loss_aux = criterion(logits_aux, target_a_var) * lam + criterion(logits_aux, target_b_var) * (1. - lam)
                loss += args.auxiliary_weight*loss_aux
        else:
            logits, logits_aux = model(input)
            loss = criterion(logits, target)
            if args.auxiliary:
                loss_aux = criterion(logits_aux, target)
                loss += args.auxiliary_weight*loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        batch_time.update(time.time() - b_start)
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if logging_mode:
            if step % args.report_freq == 0:
                end_time = time.time()
                if step == 0:
                    duration = 0
                    start_time = time.time()
                else:
                    duration = end_time - start_time
                    start_time = time.time()
                logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f Duration: %ds BTime: %.3fs', 
                                        step, objs.avg, top1.avg, top5.avg, duration, batch_time.avg)

    return top1.avg, objs.avg


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2