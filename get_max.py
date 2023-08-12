import json
import numpy as np
import os
import ipdb

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str)
parser.add_argument('--test_start', type=int)
parser.add_argument('--num', type=int)

opt = parser.parse_args()
path=opt.path

test_start=opt.test_start

with open(path,"r") as f:
    meta=json.load(f)

exps=[]
for frame in meta['frames']:
    exps.append(frame["exp_ori"])


num=opt.num
exps=np.array(exps,dtype=np.float32)

max_per=np.max(exps[:test_start],axis=0)
np.savetxt(os.path.join(os.path.dirname(path),"max_"+ str(num) +".txt"),max_per[:num])

min_per=np.min(exps[:test_start],axis=0)
np.savetxt(os.path.join(os.path.dirname(path),"min_"+ str(num) +".txt"),min_per[:num])
