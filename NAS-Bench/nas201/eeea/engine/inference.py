import logging
import eeea.utils.utils as utils
import torch
import time

def infer_cifar(args, valid_queue, model, criterion, logging_mode=False):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            target = target.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
            
            logits, _ = model(input)
            loss = criterion(logits, target)             

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            if logging_mode:
                if step % args.report_freq == 0:
                    logging.info('valid %03d %f %f %f mem %.2f MB', step, objs.avg, top1.avg, top5.avg, round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))

    return top1.avg, objs.avg

def infer_imagenet(args, valid_queue, model, criterion, logging_mode=False):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits, _ = model(input)
            loss = criterion(logits, target)

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
                logging.info('VALID Step: %03d Objs: %e R1: %f R5: %f Duration: %ds', step, objs.avg, top1.avg, top5.avg, duration)

    return top1.avg, top5.avg, objs.avg