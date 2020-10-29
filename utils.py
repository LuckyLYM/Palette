import os
import numpy as np
import torch
import shutil
import sys
import random
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.vision import VisionDataset
import torchvision.datasets as dset
import re


if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle






##################################### Custom Dataset Folder #####################################
def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images

class CustomDatasetFolder(VisionDataset):
    def __init__(self, root, loader, dirs, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):
        super(CustomDatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root, dirs)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    # change find_classes
    def _find_classes(self, dir, dirs):

        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()

        if len(dirs)!=0:
            classes=[classes[i] for i in dirs]

        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def __len__(self):
        return len(self.samples)



IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

##################################### Custom Image Folder ####################################
class CustomImageFolder(CustomDatasetFolder):
    def __init__(self, root, dirs,transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):

        super(CustomImageFolder, self).__init__(root, loader, dirs, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples




#######################################  Custom CIFAR100 dataset ####################################
class CustomCIFAR100(Dataset):
    def __init__(self,root,meta,dataset,train=True,transform=None,target_transform=None):
        self.data=[]
        self.targets=[]
        self.transform=transform
        self.target_transform=target_transform


        base_folder = 'cifar-100-python'

        path=os.path.join(root,base_folder)
        if not os.path.exists(path):
            raise RuntimeError('Dataset not found')


        if train==True:
            filepath=os.path.join(path,'train')
        else:
            filepath=os.path.join(path,'test')


        with open(filepath,'rb') as f:
            if sys.version_info[0] == 2:
                entry = pickle.load(f)
            else:
                entry = pickle.load(f, encoding='latin1')
            self.data.append(entry['data'])
            if 'labels' in entry:
                self.targets.extend(entry['labels'])
            else:
                self.targets.extend(entry['fine_labels'])
        f.close()

        # use filter to extract the dataset we want
        meta_path=os.path.join(root,meta)
        with open(meta_path,'rb') as f:
            cifar100=pickle.load(f)
            if not dataset in cifar100:
                print('invalid dataset: %s'%dataset)
                sys.exit(1)    
            select=cifar100[dataset]
        f.close()

        select.sort()
        mapper=dict()
        for index,value in enumerate(select):
            mapper[value]=index

        split=set(select)
        index=[i for i,x in enumerate(self.targets) if x in split]
        newlabel=np.array(self.targets)[index]
        # rearrange the labels
        self.targets=[mapper[i] for i in newlabel]


        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data=self.data[index]
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC


    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        img,target=self.data[index],self.targets[index]
        img=Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target



##################################### Custom CIFAR10 dataset #################################
class CustomDataset(Dataset):
  def __init__(self,root,dataset,train=True,transform=None,target_transform=None):
      self.data=[]
      self.targets=[]
      self.transform=transform
      self.target_transform=target_transform


      base_folder = 'cifar-10-batches-py'

      path=os.path.join(root,dataset)
      if not os.path.exists(path):
          raise RuntimeError('Dataset not found')


      if not os.path.isdir(path):
          if train==True:
              filepath=path
          else:
              filepath=os.path.join(root,base_folder,'test_batch')
      else:
          if train==True:
              filepath=os.path.join(path,'train')
          else:
              filepath=os.path.join(path,'test')

      with open(filepath,'rb') as f:
          if sys.version_info[0] == 2:
              entry = pickle.load(f)
          else:
              entry = pickle.load(f, encoding='latin1')
          self.data.append(entry['data'])
          if 'labels' in entry:
              self.targets.extend(entry['labels'])
          else:
              self.targets.extend(entry['fine_labels'])          
      self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
      self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

  def __len__(self):
      return len(self.data)

  def __getitem__(self,index):
      img,target=self.data[index],self.targets[index]
      img=Image.fromarray(img)

      if self.transform is not None:
          img = self.transform(img)

      if self.target_transform is not None:
          target = self.target_transform(target)

      return img, target

######################################################################################################

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt



def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)
  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t() 
  correct = pred.eq(target.view(1, -1).expand_as(pred))
  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def _data_transforms_SVHN(args):
  SVHN_MEAN = [0.5, 0.5, 0.5]
  SVHN_STD = [0.5, 0.5, 0.5]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(SVHN_MEAN, SVHN_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(SVHN_MEAN, SVHN_STD),
    ])
  return train_transform, valid_transform


def _data_transforms_tiny32(args):
  normalize = transforms.Normalize(mean=[0.4798, 0.4478, 0.3985], std=[0.2291, 0.2264, 0.2256])

  train_transforms=transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      normalize,])

  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transforms= transforms.Compose([
      transforms.ToTensor(),
      normalize,])
  return train_transforms,valid_transforms

def _data_transforms_cifar10(args):
  # data statistics
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform



def _data_transforms_FashionMNIST(args):
  # data statistics
  FMNIST_MEAN = [0.5]
  FMNIST_STD = [0.5]

  train_transform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(FMNIST_MEAN, FMNIST_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(FMNIST_MEAN, FMNIST_STD),
    ])

  return train_transform, valid_transform

def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)

def clearAuxliaryHead(model):
  keys=[]
  for key in model.keys():
    if key.startswith('auxiliary_head'):
      keys.append(key)
  for key in keys:
    model.pop(key)

  return model

def load(model,model_path,optimizer=None):
  checkpoint=torch.load(model_path)
  if 'optimizer_state_dict' not in checkpoint.keys():
    checkpoint=clearAuxliaryHead(checkpoint)
    model.load_state_dict(checkpoint)
  else:
    checkpoint=clearAuxliaryHead(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer!=None:
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])



def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)


def getLayers(model):
    m = re.search(r'\d+$', model)
    if m:
      return int(m.group())
    else:
      print('can not extract layer info from %s'%model)
      sys.exit(1)

def getClasses(dataset,meta):
    if dataset=='CIFAR_10':
        return 10
    elif dataset=='CIFAR_100':
        return 100
    elif re.match('CIFAR_10_',dataset):
        return 10
    else:
        meta_path=os.path.join('data',meta)
        f=open(meta_path,'rb')
        data= pickle.load(f)

        if not dataset in data:
            print('invalid dataset: %s'%dataset)
            sys.exit(1)    
        else:
            return len(data[dataset])


def checkClasses(dataset,classes,meta):
    real_classes=0
    if dataset=='CIFAR_10':
        real_classes=10
    elif dataset=='CIFAR_100':
        real_classes=100
    else:
        # load meta file here
        meta_path=os.path.join('data',meta)
        f=open(meta_path,'rb')
        data= pickle.load(f)

        if not dataset in data:
            print('invalid dataset: %s'%dataset)
            sys.exit(1)    
        else:
            real_classes=len(data[dataset])

    if real_classes==classes:
        return classes
    else:
        print('invalid class num %d;  %d is expected'%(classes,real_classes))
        sys.exit(1)  


def getData(dataset,dataset_meta,args,validOnly=False):
    if dataset=='CIFAR_10':
        train_transform, valid_transform = _data_transforms_cifar10(args)

        valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
        valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size,shuffle=False,pin_memory=True, num_workers=2)

        if validOnly:
            return valid_queue

        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,pin_memory=True, num_workers=2)

    elif dataset=='CIFAR_100':
        train_transform, valid_transform = _data_transforms_cifar10(args)

        valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
        valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size,shuffle=False,pin_memory=True, num_workers=2)

        if validOnly:
            return valid_queue

        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,pin_memory=True, num_workers=2)

    # subsets of CIFAR_10
    elif re.match('CIFAR_10_',dataset):

        train_transform, valid_transform = _data_transforms_cifar10(args)

        valid_data = CustomDataset(root=args.data,dataset=dataset,train=False,transform=valid_transform)
        valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size,shuffle=False,pin_memory=True, num_workers=2)

        if validOnly:
            return valid_queue

        train_data = CustomDataset(root=args.data,dataset=dataset,train=True,transform=train_transform)
        train_queue = torch.utils.data.DataLoader(
          train_data, batch_size=args.batch_size,shuffle=True,pin_memory=True, num_workers=2)

    elif re.match('CIFAR',dataset):
        train_transform, valid_transform = _data_transforms_cifar10(args)

        valid_data = CustomCIFAR100(root=args.data,meta=dataset_meta,dataset=dataset,train=False,transform=valid_transform)
        valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size,shuffle=False,pin_memory=True, num_workers=2)

        if validOnly:
            return valid_queue

        train_data = CustomCIFAR100(root=args.data,meta=dataset_meta,dataset=dataset,train=True,transform=train_transform)
        train_queue = torch.utils.data.DataLoader(
          train_data, batch_size=args.batch_size,shuffle=True,pin_memory=True, num_workers=2)

    elif re.match('tiny',dataset):
        train_transform, valid_transform = _data_transforms_tiny32(args)

        meta=os.path.join('data',dataset_meta)

        f=open(meta,'rb')
        meta=pickle.load(f)
        dirs=meta[dataset]

        validdir = os.path.join(args.data,'tiny','val')
        valid_data = CustomImageFolder(validdir,dirs,valid_transform)
        valid_queue = torch.utils.data.DataLoader(
          valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

        if validOnly:
          return valid_queue

        traindir = os.path.join(args.data,'tiny','train')
        train_data =CustomImageFolder(traindir,dirs,train_transform)
        train_queue = torch.utils.data.DataLoader(
          train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
        
    elif re.match('few',dataset):
        train_transform, valid_transform = _data_transforms_tiny32(args)

        meta=os.path.join('data',dataset_meta)

        f=open(meta,'rb')
        meta=pickle.load(f)
        dirs=meta[dataset]

        validdir = os.path.join(args.data,'few','val')
        valid_data = CustomImageFolder(validdir,dirs,valid_transform)
        valid_queue = torch.utils.data.DataLoader(
          valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

        if validOnly:
          return valid_queue

        traindir = os.path.join(args.data,'few','train')
        train_data =CustomImageFolder(traindir,dirs,train_transform)
        train_queue = torch.utils.data.DataLoader(
          train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    else:
        print('Invalid dataset %s'%dataset)
        sys.exit(1)

    return train_queue,valid_queue


def getTransferData(path,epochs,lastepoch=0,max_acc=True):

    time=0
    acc=0
    if not os.path.exists(path) or epochs==0:
        return time,acc

    f=open(path,"r")
    lines=f.readlines()

    #print('epochs %d lastepoch %d len(lines) %d'%(epochs,lastepoch,len(lines)))
    if epochs>len(lines):
        print('invalid index:%d   len:%d'%(epochs,len(lines)))
        print('path: %s'%path)
        sys.exit(1)
        


    line=lines[epochs-1]
    line=line.split()
    for index,seg in enumerate(line):
        if seg=='time:':
            time=float(line[index+1])
            break
    
    acc=float(line[-1])/100
    if time==0:
        print('can not find time info in path %s'%path)
        sys.exit(1)

    if lastepoch!=0:
        line=lines[lastepoch-1]
        line=line.split()
        for index,seg in enumerate(line):
            if seg=='time:':
                lasttime=float(line[index+1])
                break
        time=time-lasttime


    if max_acc==True:
        best_acc=0
        for i in range(epochs):
            acc=float(lines[i].split()[-1])/100
            if acc>best_acc:
                best_acc=acc
        acc=best_acc
    
    return time,acc
