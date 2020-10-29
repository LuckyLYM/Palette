import os
import os.path
import numpy as np
import sys
import random

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle




def generateDataset():
    data=dict()
    for i in range(1,101):
        class_num=random.randint(20,90)  # 20-90
        chosen_class=random.sample(list(range(100)),class_num)
        data['CIFAR'+str(i)]=chosen_class

    file_path=os.path.join('CIFAR_meta')
    fw=open(file_path,'wb')
    pickle.dump(data,fw,-1)  

def generateModels():
    models=[]
    layers=[8,10,12]
    for i in range(1,101):
        layer=layers[random.randint(0,2)]
        models.append(layer)

    file_path=os.path.join('models.meta')
    fw=open(file_path,'wb')
    pickle.dump(models,fw,-1)     

generateDataset()
generateModels()