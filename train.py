import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import time
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model import NetworkCIFAR as Network
import genotypes


parser = argparse.ArgumentParser("train model")
parser.add_argument('--data', type=str, default='data', help='location of the data corpus')   
parser.add_argument('--layers', type=int, default=10, help='total number of layers')         
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')        
parser.add_argument('--class_num', type=int, default=-1, help='num of classes')              
parser.add_argument('--batch_size', type=int, default=96, help='batch size')  # here
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')  
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--arch', type=str, default='DARTS', help='the previously stored genotype')
parser.add_argument('--stem', type=str, default='None', help='name of the stem dataset')    
parser.add_argument('--dataset', type=str, default='None', help='name of the dataset')    
parser.add_argument('--meta',type=str,default='None',help='the version of dataset meta file')


args = parser.parse_args()


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


RootDir=args.stem


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  start=time.time()

  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  cudnn.enabled=True
  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()



  genotype = eval("genotypes.%s" % args.arch)


  exp_path=os.path.join(RootDir,'EXP',args.dataset)
  if not os.path.exists(exp_path):
    os.makedirs(exp_path)
  exp_path=os.path.join(exp_path,args.arch+str(args.layers)) 
  f=open(exp_path,"w")

  model_path=os.path.join(RootDir,'models',args.dataset)
  if not os.path.exists(model_path):
    os.makedirs(model_path)
  model_name=args.arch+str(args.layers)+'.pt'
  model_path=os.path.join(model_path,model_name)



  model = Network(args.init_channels, utils.getClasses(args.dataset,args.meta), args.layers, args.auxiliary, genotype)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)


  train_queue,valid_queue=utils.getData(args.dataset,args.meta,args)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

  best_acc=0
  epochs=args.epochs
  for epoch in range(args.epochs):
    logging.info('epoch %d lr %e', epoch+1, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    logging.info('epoch %d train_acc %f', epoch+1,train_acc)
    scheduler.step()

    top1_acc, top5_acc, valid_obj = infer(valid_queue, model, criterion)
    duration=time.time()-start

    # new feature
    # save the best possible model during the search process
    if top1_acc>best_acc:
      best_acc=top1_acc
      #store model parameters
      torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
      },model_path)      

    f.write('epoch: %d  valid_obj: %f time: %.6f top5_acc: %f top1_acc: %f \n'%(epoch+1,valid_obj,duration,top5_acc,top1_acc))
    logging.info('epoch: %d  valid_obj: %f time: %.6f top5_acc: %f top1_acc: %f \n'%(epoch+1,valid_obj,duration,top5_acc,top1_acc))

  f.close()



def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda()

    optimizer.zero_grad()
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
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    with torch.no_grad():
        input = Variable(input).cuda()
        target = Variable(target).cuda()

    logits, _ = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg,top5.avg, objs.avg


if __name__ == '__main__':
  main() 