import os
import sys
import time
import glob
import numpy as np
import random
import torch
import utils
import time
import logging
import argparse
import torch.nn as nn
import torch.utils
import datetime
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import math
import pickle
import re
import copy


from torch.autograd import Variable
from model import NetworkCIFAR as Network
from model import NetworkEnsemble
from GP import GPUCB, Environment
import genotypes


parser = argparse.ArgumentParser("model selection")
parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=30, help='num of training epochs')        
parser.add_argument('--batch_size', type=int, default=96, help='batch size')  # here
parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=200, help='report frequency')
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
parser.add_argument('--max_acc', action='store_true',default=False, help='whether maximum acc')
parser.add_argument('--ensemble_size',type=int,default=3,help='maximum number of ensemble size')
parser.add_argument('--transfer', type=str, default='fine_tune', help='transfer method')
parser.add_argument('--mode',type=str,default='simulation',help='experiment mode')
parser.add_argument('--discount',type=int,default='3',help=' a parameter used in discount selection strategy')
parser.add_argument('--selection_strategy',type=str,default='random',help='the model selecton strategy')
parser.add_argument('--ensemble_strategy',type=str,default='simple_voting',help='the model selecton strategy')
parser.add_argument('--source_meta',type=str,default='None',help='the version of dataset meta file')
parser.add_argument('--target_meta',type=str,default='None',help='the version of dataset meta file')
parser.add_argument('--target_stem',type=str,default='None',help='the name of the target stem dataset')
parser.add_argument('--source_stem',type=str,default='None',help='the name of the source stem dataset')
parser.add_argument('--target_dataset', type=str, default='None', help='the name of target') 
parser.add_argument('--source_models', type=str, default='None', help='the name of target') 
parser.add_argument('--EXP', type=str, default='None', help='specific experiments') 
parser.add_argument('--stem_list',type=str,default='None',help='the dirs of target experiments')




args = parser.parse_args()
sourceDir=args.source_stem
targetDir=args.target_stem
now=datetime.datetime.now()
now=datetime.datetime.strftime(now,'%m-%d %H:%M:%S')

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


  if args.mode=='simulation':
    source_path=os.path.join(args.target_stem,'EXP',target, source_model+"_"+source_dataset)
    if not os.path.exists(source_path):
      logging.info('failed in simulation mode: %s does not exist'%source_path)
      sys.exit(1)
    else:
      return utils.getTransferData(source_path,epochs+lastepoch,lastepoch,max_acc=args.max_acc)
      
  start=time.time()
  oldTime=0

  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  cudnn.enabled=True
  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  if lastepoch==0:
    source_path=os.path.join(args.source_stem,'models',source_dataset,source_model+".pt")
  else:
    source_path=os.path.join(args.target_stem,'models',target,now,source_model+"_"+source_dataset+".pt")

  if not os.path.exists(source_path):
    logging.info('path: %s'%source_path)
    logging.info('source model does not exist')
    sys.exit(1)

  target_path=os.path.join(args.target_stem,'models',target,now)
  if not os.path.exists(target_path):
    os.makedirs(target_path)
  target_path=os.path.join(target_path,source_model+"_"+source_dataset+".pt")


  exp_path=os.path.join(args.target_stem,'EXP',target,now)
  if not os.path.exists(exp_path):
    os.makedirs(exp_path)
  exp_path=os.path.join(exp_path,source_model+"_"+source_dataset) 
  oldTime,__=utils.getTransferData(exp_path,lastepoch,max_acc=args.max_acc)
  f=open(exp_path,"a+")


  genotype = eval("genotypes.%s" % args.arch)
  if lastepoch==0: 
    model = Network(args.init_channels, utils.getClasses(source_dataset,args.source_meta), utils.getLayers(source_model), args.auxiliary, genotype)
    optimizer=getOptimizer(model)
    utils.load(model,source_path)   

    num_ftrs = model.classifier.in_features
    model.classifier=nn.Linear(num_ftrs,utils.getClasses(target,args.target_meta))
  
  else: 
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
  f.close()

  torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict' : optimizer.state_dict(),
  },target_path)

  torch.cuda.empty_cache()

  return time.time()-start,top1_acc


############################################ ensemble strategy ########################################
def greedyForwardSelection(models,target):

  # assume the input models have already beed sorted according to validation accuracy
  start=time.time()
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  cudnn.enabled=True
  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  genotype = eval("genotypes.%s" % args.arch)
  valid_queue=utils.getData(target,args.target_meta,args,validOnly=True)
  logit_list=[]
  acc_list=[]
  model_size_list=[]


  counter=0


  for pair in models:
    source_dataset=pair[0]
    source_model=pair[1]

    if args.mode=='simulation':
      target_path=os.path.join(args.target_stem,'models',target)
    else:
      target_path=os.path.join(args.target_stem,'models',target,now)


    target_path=os.path.join(target_path,source_model+"_"+source_dataset+".pt")
    target_trial_path=target_path+'_'+str(args.epochs)
    if os.path.exists(target_trial_path):
      target_path=target_trial_path    


    model = Network(args.init_channels, utils.getClasses(target,args.target_meta), utils.getLayers(source_model), args.auxiliary, genotype)
    utils.load(model,target_path)   
    model.drop_path_prob = args.drop_path_prob
    model = model.cuda()

    model_size=utils.count_parameters_in_MB(model)
    model_size_list.append(model_size)

    logging.info('cal logits for model %s'%str(pair))
    logits,acc=get_logit(valid_queue,model,criterion) 
    acc_list.append(acc)

    torch.cuda.empty_cache()
    logit_list.append(logits)

  K=len(models)
  best_acc=acc_list[0]
  best_digit=copy.deepcopy(logit_list[0])
  best_current_digit=[]

  model_index=[0]
  cadidate_index=[i for i in range(1,K)]


  t1=time.time()-start

  logging.info('cal logit time: %f'%t1)

  while True:
    best_candidate=-1
    for index in cadidate_index:
      ## combine with best digit
      stem=copy.deepcopy(best_digit)
      temp=logit_list[index]
      top1 = utils.AvgrageMeter()
      top5 = utils.AvgrageMeter()
      for step, (input, target) in enumerate(valid_queue):
          stem[step]=stem[step]+temp[step]
          prec1, prec5 = utils.accuracy(stem[step], target, topk=(1, 5))
          batchsize = input.size(0)
          top1.update(prec1.data.item(), batchsize)
          top5.update(prec5.data.item(), batchsize)
      if top1.avg>best_acc:
        best_acc=top1.avg
        best_candidate=index
        best_current_digit=stem

      counter=counter+1
      t=time.time()-start
      logging.info('trail: %d time: %f'%(counter,t))  

    if best_candidate!=-1:
      model_index.append(best_candidate)
      cadidate_index.remove(best_candidate)
      best_digit=best_current_digit

      size=len(model_index)
      logging.info('###### Ensemble %d models: %s ######'%(size,str(model_index)))
      logging.info('Top1 acc: %f'%best_acc)

    if best_candidate==-1 or len(cadidate_index)==0:
      if best_candidate==-1:
        reason='no better candidate'
      else:
        reason='run out of candidates'
      logging.info('Greedy forward selection terminated: %s'%reason)
      break


  total_model_size=0

  for index in model_index:
    total_model_size=total_model_size+model_size_list[index]

  duration=time.time()-start
  return duration, best_acc, len(model_index) ,total_model_size


def simpleVoting(models,target):
  start=time.time()

  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  cudnn.enabled=True
  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  genotype = eval("genotypes.%s" % args.arch)
  valid_queue=utils.getData(target,args.target_meta,args,validOnly=True)

  K=len(models)

  logit_list=[]

  total_model_size=0

  for pair in models:
    source_dataset=pair[0]
    source_model=pair[1]

    if args.mode=='simulation':
      target_path=os.path.join(args.target_stem,'models',target)
    else:
      target_path=os.path.join(args.target_stem,'models',target,now)
    
    target_path=os.path.join(target_path,source_model+"_"+source_dataset+".pt")
    target_trial_path=os.path.join(target_path,'_'+str(args.epochs))
    if os.path.exists(target_trial_path):
      target_path=target_trial_path



    model = Network(args.init_channels, utils.getClasses(target,args.target_meta), utils.getLayers(source_model), args.auxiliary, genotype)
    utils.load(model,target_path)   
    model.drop_path_prob = args.drop_path_prob
    model = model.cuda()

    model_size=utils.count_parameters_in_MB(model)
    total_model_size=total_model_size+model_size

    logging.info('cal logits for model %s'%str(pair))
    logits,acc=get_logit(valid_queue,model,criterion) 

    torch.cuda.empty_cache()
    logit_list.append(logits)


  stem=logit_list[0]
  best_acc=0
  average_acc=0
  for n in range(2,K+1):  
    print('****************Ensemble %d models******************'%n)
    temp=logit_list[n-1]
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    for step, (input, target) in enumerate(valid_queue):
        stem[step]=stem[step]+temp[step]
        prec1, prec5 = utils.accuracy(stem[step], target, topk=(1, 5))
        batchsize = input.size(0)
        top1.update(prec1.data.item(), batchsize)
        top5.update(prec5.data.item(), batchsize)

    average_acc=top1.avg

    if top1.avg>best_acc:
        best_acc=top1.avg
    logging.info('Simple Soft Voting: valid acc %f', top1.avg)
  

  duration=time.time()-start
  return duration, average_acc, len(models), total_model_size


def weightedVoting(models,target):
  start=time.time()

  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  cudnn.enabled=True
  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  genotype = eval("genotypes.%s" % args.arch)

  total_model_size=0

  modellist=[]
  for pair in models:
    source_dataset=pair[0]
    source_model=pair[1]

    if args.mode=='simulation':
      target_path=os.path.join(args.target_stem,'models',target)
    else:
      target_path=os.path.join(args.target_stem,'models',target,now)

    target_path=os.path.join(target_path,source_model+"_"+source_dataset+".pt")
    target_trial_path=os.path.jpin(target_path,'_'+str(args.epochs))
    if os.path.exists(target_trial_path):
      target_path=target_trial_path

    model = Network(args.init_channels, utils.getClasses(target,args.target_meta), utils.getLayers(source_model), args.auxiliary, genotype)
    utils.load(model,target_path)   
    model.drop_path_prob = args.drop_path_prob

    model.cuda()

    model_size=utils.count_parameters_in_MB(model)
    total_model_size=total_model_size+model_size

    logging.info('load model %s'%str(pair))

    modellist.append(model)

  ensemble=NetworkEnsemble(modellist,utils.getClasses(target,args.target_meta))
  ensemble.cuda()

  optimizer = torch.optim.SGD(
      ensemble.classifier.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)


  train_queue,valid_queue=utils.getData(target,args.target_meta,args)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

  
  epochs=args.epochs
  best_acc=0
  for epoch in range(args.epochs):
    logging.info('epoch %d lr %e', epoch+1, scheduler.get_lr()[0])

    train_acc, train_obj = trainEnsemble(train_queue, ensemble, criterion, optimizer)
    logging.info('epoch %d train_acc %f', epoch+1,train_acc)
    scheduler.step()

    top1_acc, top5_acc, valid_obj = inferEnsemble(valid_queue, ensemble, criterion)
    duration=time.time()-start
    if top1_acc>best_acc:
      best_acc=top1_acc

    logging.info('epoch: %d  valid_obj: %f time: %.6f top5_acc: %f top1_acc: %f \n'%(epoch+1,valid_obj,duration,top5_acc,top1_acc))


  duration=time.time()-start
  return duration, best_acc




############################################# Selection Strategy ##########################################
def normalize(a):
  a=np.array(a)
  Min=np.min(a)
  Max=np.max(a)
  return (a-Min)/(Max-Min)


def getStats(models,target):
  path=os.path.join('data',args.source_meta)
  f=open(path,'rb')
  dataset=pickle.load(f)
  x=[]
  y=[]
  for pair in models:
    source_dataset=pair[0]
    source_model=pair[1]
    length=len(dataset[source_dataset])
    layer=utils.getLayers(source_model)
    x.append(length)
    y.append(layer)
  x=normalize(x)
  y=normalize(y)

  rewards=[]
  cost=[]

  for pair in models:
    source_dataset=pair[0]
    source_model=pair[1]
    target_model=source_model+'_'+source_dataset
    path=os.path.join(args.target_stem,'EXP',target,target_model)

    acc_list=[]
    time_list=[]

    f=open(path,"r")
    lines=f.readlines()
    for line in lines:
      time=0
      acc=0
      line=line.split()
      for index,seg in enumerate(line):
        if seg=='time:':
          time=float(line[index+1])
          break
      acc=float(line[-1])/100
      time_list.append(time)
      acc_list.append(acc)

    rewards.append(acc_list)
    cost.append(time_list)

  return x,y,rewards,cost


def selectTopK(models,acc,K):
    acc=np.array(acc)
    index=np.argsort(-acc)
    selectedModels=[ models[index[i]] for i in range(K)]
    selectedAcc=[acc[index[i]] for i in range(K)]
    return selectedModels,selectedAcc


def randomSelection(models,target,K):
    transferTime=0
    transferAcc=[]
    ensembleTime=0
    ensembleAcc=0
    selectedModels=[]
    if len(models)<=K:
        selectedModels=models
    else:
        selectedModels=random.sample(models,K)

    for pair in selectedModels:
        time,acc=transfer(pair,target,args.epochs)
        transferTime=transferTime+time
        transferAcc.append(acc)


    logging.info('################## experiments summary ##################')
    logging.info('selection strategy: Random')
    logging.info('selected models %s'%str(selectedModels))
    logging.info('fine-tuning acc %s'%str(transferAcc))
    logging.info('fine-tuning time %f'%transferTime)

    return selectedModels,transferAcc,[transferTime]



def GPSelection(models,target,K):

    x,y,rewards,cost=getStats(models,target)
    env = Environment(rewards,cost)
    agent = GPUCB(np.array([x, y]), env, K)
    selectionTime,selectedIndex, samples=agent.selection(K)
    selectedModels=[models[i] for i in selectedIndex]

    transferAcc=[]
    transferTime=0

    for pair in selectedModels:
        time,acc=transfer(pair,target,args.epochs-1,1)
        transferTime=transferTime+time
        transferAcc.append(acc)

    logging.info('################## experiments summary ##################')
    logging.info('number of samples: %d'%samples)
    logging.info('selection strategy: GP-UCB')
    logging.info('selection time %s'%str(selectionTime))
    logging.info('selected models %s'%str(selectedModels))
    logging.info('fine-tuning acc %s'%str(transferAcc))
    logging.info('fine-tuning time %f'%transferTime)

    return selectedModels,transferAcc,[selectionTime,transferTime]



def oneShotSelection(models,target,K):

    oneShotTime=0
    oneShotAcc=[]

    for pair in models:
        time,acc=transfer(pair,target,1)
        oneShotTime=oneShotTime+time
        oneShotAcc.append(acc)

    selectedModels,__=selectTopK(models,oneShotAcc,K)

    transferTime=0
    transferAcc=[]
    for pair in selectedModels:
        time,acc=transfer(pair,target,args.epochs-1,1)
        transferTime=transferTime+time
        transferAcc.append(acc)

    logging.info('################## experiments summary ##################')
    logging.info('selection strategy: OneShot')
    logging.info('selection time: %f'%oneShotTime)
    logging.info('selected models %s'%str(selectedModels))
    logging.info('fine-tuning acc %s'%str(transferAcc))
    logging.info('fine-tuning time %f'%transferTime)

    return selectedModels,transferAcc,[oneShotTime,transferTime]


def transferAllSelection(models,target,K):

    transferTime=0
    transferAcc=[]
    for pair in models:
        time,acc=transfer(pair,target,args.epochs)
        transferTime=transferTime+time
        transferAcc.append(acc)

    selectedModels,selectedAcc=selectTopK(models,transferAcc,K)

    logging.info('################## experiments summary ##################')
    logging.info('selection strategy: FA')
    logging.info('selected models %s'%str(selectedModels))
    logging.info('fine-tuning acc %s'%str(selectedAcc))
    logging.info('fine-tuning time %f'%transferTime)

    return selectedModels,selectedAcc,[transferTime]


def halvingSeection(models,target,K,double=True):
    n=len(models)
    selectedModels=models
    nIter=int(math.log(n/K,2))

    B=1
    lastepoch=0

    banditTime=0
    transferTime=0
    transferAcc=[]

    for k in range(nIter):

        banditAcc=[]
        epochs=B
        if double==True:
            B=B*2

        if epochs+lastepoch>=args.epochs:
            epochs=args.epochs-lastepoch

        for pair in selectedModels:
            source_dataset=pair[0]
            source_model=pair[1]
            time,acc=transfer(pair,target,epochs,lastepoch) 
            banditTime=banditTime+time
            banditAcc.append(acc)

        selectedModels,transferAcc=selectTopK(selectedModels,banditAcc,int(len(selectedModels)/2))
        lastepoch=lastepoch+epochs


        if lastepoch>=args.epochs:
            break

    if len(selectedModels)>K:
        selectedModels,transferAcc=selectTopK(selectedModels,transferAcc,K)


    if lastepoch<args.epochs:
        transferAcc=[]
        epochs=args.epochs-lastepoch
        for pair in selectedModels:
            source_dataset=pair[0]
            source_model=pair[1]
            time,acc=transfer(pair,target,epochs,lastepoch) 
            transferTime=transferTime+time
            transferAcc.append(acc)    

    logging.info('################## experiments summary ##################')
    logging.info('selection strategy: SH')
    logging.info('selection time: %f'%banditTime)
    logging.info('selected models %s'%str(selectedModels))
    logging.info('fine-tuning acc %s'%str(transferAcc))
    logging.info('fine-tuning time %f'%transferTime)

    return selectedModels,transferAcc,[banditTime,transferTime]



def getSourceModels():
    data=args.source_models
    model_path=os.path.join('data','models.meta')
    f=open(model_path,'rb')
    models=pickle.load(f)

    base=[]
    for i in range(1,100+1):
        dataset=data+str(i)
        layer=models[i-1]
        model=args.arch+str(layer)
        t=(dataset,model)
        base.append(t)

    return base

def topKSelection(models,target,K):

  if args.selection_strategy=='Random':
    return randomSelection(models,target,K)
  elif args.selection_strategy=='OneShot':
    return oneShotSelection(models,target,K)
  elif args.selection_strategy=='SH':
    return halvingSeection(models,target,K)
  elif args.selection_strategy=='FA':
    return transferAllSelection(models,target,K)
  elif args.selection_strategy=='GP':
    return GPSelection(models,target,K)

def ensemble(models,target):
  if args.ensemble_strategy=='simple_voting':
    return simpleVoting(models,target)
  elif args.ensemble_strategy=='weighted_voting':
    return weightedVoting(models,target)
  elif args.ensemble_strategy=='greedy_forward':
    return greedyForwardSelection(models,target)
  else:
    return 0,0



def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

    models=getSourceModels()
    target=args.target_dataset
    K=args.ensemble_size
    selectedModels,transferAcc,time=topKSelection(models,target,K)
    ensembleTime,ensembleAcc,_,_=ensemble(selectedModels,target)


def get_logit(valid_queue, model, criterion):
  model.eval()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  logit=[]

  for step, (input, target) in enumerate(valid_queue):
    with torch.no_grad():
        input = Variable(input).cuda()
        target = Variable(target).cuda()

    logits, _ = model(input)  

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    logits=logits.cpu().data
    logit.append(logits)  

  return logit, top1.avg

def trainEnsemble(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda()

    optimizer.zero_grad()
    logits= model(input)
    loss = criterion(logits, target)
    
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


def inferEnsemble(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    with torch.no_grad():
        input = Variable(input).cuda()
        target = Variable(target).cuda()

    logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, top5.avg, objs.avg


def infer(valid_queue, model, criterion,split=1):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  nEpoch=0
  dataNum=len(valid_queue.dataset)
  batchSize=valid_queue.batch_size
  validNum=int(dataNum*split)
  nEpoch=math.ceil(validNum/batchSize)

  for step, (input, target) in enumerate(valid_queue):
    if step==nEpoch:
      break

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

  return top1.avg, top5.avg, objs.avg



def train(train_queue, model, criterion, optimizer, split=1):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  nEpoch=0
  dataNum=len(train_queue.dataset)
  batchSize=train_queue.batch_size
  trainNum=int(dataNum*split)
  nEpoch=math.ceil(trainNum/batchSize)

  for step, (input, target) in enumerate(train_queue):
    if step==nEpoch:
      break

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


if __name__ == '__main__':
  main() 
