from __future__ import division
import pickle
import math
import numpy as np
import sys

"""the purpose of this script is to generate
5-fold train-test set of different xor obfuscated text
that contains different ciphers"""

def bin_dec(nb):
    # binarize a 
    # sample token
    ln = 8
    vec = np.zeros((ln,), dtype=np.int)
    binkey = bin(int(nb))
    bef, af = binkey.split("b")   
    for j, ltr in enumerate(af[::-1]):
        vec[-j-1]=int(ltr)
    return vec

def bin_it(lbl, num):
    # binarize a label
    xor, key = lbl.split("_")
    binkey = bin(int(key))
    bef, af = binkey.split("b")
    vec = np.zeros((num,), dtype=np.int)
    for j, ltr in enumerate(af[::-1]):
        vec[-j-1]=int(ltr)
    return vec

# hist the data to labels
path2synth = sys.argv[1]
path2fold = sys.argv[2]
settype = sys.argv[3]

# reduce or add to 100 nodes
nul = np.zeros(8)
X = []
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
        label = bin_it(nest[-1], num=8)
        words = tokens[:-n_nest-1]
        new_ws = []
        # if empty
        if len(words)==0:
            continue
        for e, word in enumerate(words):
            dec = 0
            # convert to decimal
            for f, ltr in enumerate(word[::-1]):
                if ltr.isalpha():
                    ltr = ord(ltr)-87
                else:
                    ltr = int(ltr)	
                dec+= ltr*math.pow(16,f)    
            dec = bin_dec(dec)
            new_ws.append(dec)   
        if len(new_ws)<100:
            #print "small"
            for l in xrange(100-len(new_ws)):
                new_ws.append(nul)         
        elif len(new_ws)>100:
            new_ws = new_ws[:100]
        n_shard.append(new_ws)
        nests.append(nest)
        labels.append(label)
    the_dict = { "shard": n_shard, "labels": labels, "nests": nests}	    
    # pickle fold or separatly 
    with open(path2fold+"/fold_"+str(fold)+"_"+settype, 'wb') as handle:
        pickle.dump(the_dict, handle)
