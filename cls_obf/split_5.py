from __future__ import division
import pickle
import math
import numpy as np
import sys

"""the purpose of this script is to generate 
5-fold train-test set of different obfuscations
methods: xor, gzip, base64 and plaintext"""

def lb2vec(lbl):
    vc = np.zeros(4,dtype=int)
    if lbl=="gz":
	vc[0]=1
    elif lbl=="base64":
	vc[1]=1
    elif lbl=="plaintext":
	vc[2]=1
    elif "xor" in lbl:
	vc[3]=1
    return vc

# hist the data to labels
path2synth = sys.argv[1]
path2fold = sys.argv[2]
settype = sys.argv[3]

xor = 0
gz = 0
b64 = 0
txt = 0

if 0:
    # confirm uniform dist historgram
    for line in open(path2synth, "r").readlines():
     	  label = line.rsplit()
    	  if "xor" in label:
	          xor+=1
    	  elif "gz"==label:
 	          gz+=1
    	  elif "base64"==label:
	          b64+=1
    	  else:
	          txt+=1 

# split data to 5-folds and accordignly 5 pickles
X = []
max_w = 0
for line in open(path2synth, "r").readlines():
    X.append(line)
for fold in xrange(5):
    shard = X[fold::5]
    n_shard = []
    labels = []
    nests = []
    # a line is a sample of tokens and a label
    for line in shard:
        tokens = line.split()
        n_nest = int(tokens[-1])
        nest = tokens[-n_nest-1:-1]
        label = lb2vec(nest[-1])
        words = tokens[:-n_nest-1]
        new_ws = []
        # if empty
        if len(words)<3:
            continue
        if len(words)>100:
            words = words[:100]
        for word in words:
            dec = 0
            # convert to decimal
            for f, ltr in enumerate(word[::-1]):
                if ltr.isalpha():
                    ltr = ord(ltr)-87
                else:
                    ltr = int(ltr)	
                dec+= ltr*math.pow(16,f)
            new_ws.append(int(dec))
        # convert to hot rep.
        n_shard.append(new_ws)
        nests.append(nest)
        labels.append(label)
    the_dict = { "shard": n_shard, "labels": labels, "nests": nests}
    # pickle fold or separatly 
    with open(path2fold+"/fold_"+str(fold)+"_"+settype, 'wb') as handle:
        pickle.dump(the_dict, handle)
