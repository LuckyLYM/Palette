import os
import os.path
import numpy as np
import sys
import random

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


data=dict()


for i in range(1,101):
    class_num=random.randint(20,90)
    chosen_class=random.sample(list(range(150)),class_num)
    print(chosen_class)
    data['tiny'+str(i)]=chosen_class
    file_path=os.path.join('tiny_meta')
    fw=open(file_path,'wb')
    pickle.dump(data,fw,-1)    