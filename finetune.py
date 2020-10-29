import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import time
import re

from torch.autograd import Variable
from model import NetworkCIFAR as Network
import genotypes


parser = argparse.ArgumentParser("fine-tune")
parser.add_argument('--data', type=str, default='data', help='location of the data corpus')    # should modify
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=200, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=30, help='num of training epochs')        # use less epochs
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--arch', type=str, default='DARTS', help='the previously stored genotype')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--save_frequency', type=int, default=1000, help='save model frequency')


parser.add_argument('--target_stem',type=str,default='None',help='the name of the target stem dataset')
parser.add_argument('--source_stem',type=str,default='None',help='the name of the source stem dataset')

parser.add_argument('--target_dataset', type=str, default='None', help='the name of target') 
parser.add_argument('--source_dataset', type=str, default='None', help='the name of target') 

parser.add_argument('--source_meta',type=str,default='None',help='the version of dataset meta file')
parser.add_argument('--target_meta',type=str,default='None',help='the version of dataset meta file')

parser.add_argument('--source_model',type=str,default='DARTS',help='name of the source model')
parser.add_argument('--transfer', type=str, default='fine_tune', help='transfer method')


args = parser.parse_args()


# python finetune.py --source_meta=CIFAR100_v2 --target_meta=few_meta --source_stem=CIFAR --target_stem=few  --source_dataset=CIFAR1 --target_dataset=few10 --source_model=DARTS10
sourceDir=args.source_stem
targetDir=args.target_stem



log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def getOptimizer(model):
  if args.transfer=='fine_tune':
    optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)
  elif args.transfer=='extractor':
    optimizer = torch.optim.SGD(
      model.classifier.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)   
  else:
    logging.info('invalid transfer method')
    logging.info('only support fine_tune and extractor now')
    sys.exit(1)

  return optimizer


def transfer(source,target,epochs,lastepoch=0):

  source_dataset=source[0]
  source_model=source[1]

  start=time.time()
  oldTime=0

  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  cudnn.enabled=True
  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  if lastepoch==0:
    source_path=os.path.join(sourceDir,'models',source_dataset,source_model+".pt")
  else:
    source_path=os.path.join(targetDir,'models',target,source_model+"_"+source_dataset+".pt")

  if not os.path.exists(source_path):
    logging.info('path: %s'%source_path)
    logging.info('source model: %s does not exist'%source_path)
    sys.exit(1)

  # target model path                           
  target_path=os.path.join(targetDir,'models',target)
  if not os.path.exists(target_path):
    os.makedirs(target_path)
  target_path=os.path.join(target_path,source_model+"_"+source_dataset+".pt")

  # training log data                          
  exp_path=os.path.join(targetDir,'EXP',target)
  if not os.path.exists(exp_path):
    os.makedirs(exp_path)
  exp_path=os.path.join(exp_path,source_model+"_"+source_dataset) 
  oldTime,__=utils.getTransferData(exp_path,lastepoch)
  if lastepoch==0:
    f=open(exp_path,"w")
  else:
    f=open(exp_path,"a+")    


  genotype = eval("genotypes.%s" % args.arch)
  if lastepoch==0: # target model does not exists
    # build target model
    model = Network(args.init_channels, utils.getClasses(source_dataset,args.source_meta), utils.getLayers(source_model), args.auxiliary, genotype)
    optimizer=getOptimizer(model)
    utils.load(model,source_path)   

    num_ftrs = model.classifier.in_features
    model.classifier=nn.Linear(num_ftrs,utils.getClasses(target,args.target_meta))
  
  else: # target model is already created
    model = Network(args.init_channels, utils.getClasses(target,args.target_meta), utils.getLayers(source_model), args.auxiliary, genotype)
    optimizer=getOptimizer(model)
    utils.load(model,source_path,optimizer)   



  model = model.cuda()
  model_size=utils.count_parameters_in_MB(model)
  logging.info("param size = %fMB", model_size)

  train_queue,valid_queue=utils.getData(target,args.target_meta,args)

  
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, float(args.epochs), last_epoch=-1 if lastepoch==0 else lastepoch)
  
  top1_acc=0

  epochs=lastepoch+epochs
  for epoch in range(lastepoch,epochs):
    logging.info('epoch %d lr %e', epoch+1, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    logging.info('epoch %d train_acc %f', epoch+1,train_acc)
    scheduler.step()

    top1_acc, top5_acc, valid_obj = infer(valid_queue, model, criterion)
    duration=time.time()-start+oldTime
    f.write('epoch: %d  valid_obj: %f time: %.6f top5_acc: %f top1_acc: %f \n'%(epoch+1,valid_obj,duration,top5_acc,top1_acc))
    logging.info('epoch: %d  valid_obj: %f time: %.6f top5_acc: %f top1_acc: %f \n'%(epoch+1,valid_obj,duration,top5_acc,top1_acc))

    if (epoch+1)%args.save_frequency==0:
      nepoch=epoch+1
      torch.save({
        'epoch': nepoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
      },target_path+'_'+str(nepoch))


  f.close()

  torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict' : optimizer.state_dict(),
  },target_path)

  torch.cuda.empty_cache()

  return time.time()-start,top1_acc


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  source_dataset=args.source_dataset
  source_model=args.source_model
  source=(source_dataset,source_model)

  target_dataset=args.target_dataset
  epochs=args.epochs

  transfer(source,target_dataset,epochs)
 


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