from __future__ import division
import numpy as np
import pdb 

def align_w_out(ve1, ve2, ve3, ve4, ve5, ve6, ve7, ve8):
  # align the different classifiers prediction to form a key
  raw = []
  for (v1, v2, v3, v4, v5, v6, v7, v8) in zip(ve1, ve2, ve3, ve4, ve5, ve6, ve7, ve8):
    a = np.array([v1, v2, v3, v4, v5, v6, v7, v8])
    raw.append(a)
  raw = np.array(raw)
  return raw

def concatit(X, y):
  # concatenate the input to the last prediction
  # to makee it recurrent
  merge = []
  for x,label in zip(X,y):
    # turn to 7 bit
    n_label = [0]*8
    if int(label)==1:
      n_label[0]=1  
    x = np.concatenate((x,[n_label]), axis=0)
    merge.append(x)
  return np.array(merge)

def acc(y_true, y_hat):
  # accuracy of predictions on a classifier level (bit)
  y_hat = np.array(y_hat)
  y_true = np.array(y_true)
  y_hat = (y_hat>0.5).astype('int32')
  y_true = y_true.astype('int32')
  result = y_true-y_hat
  correct = len(np.where(result==0)[0])
  incorrect = len(result)-correct
  print "incorrect - ", incorrect, np.round(incorrect*100/len(result),decimals=2)
  print "  correct - ", correct, np.round(correct*100/len(result),decimals=2)

def overall_acc(y_true, y_hat):
  # accuracy overall
  corr = 0
  y_hat = (y_hat>0.5).astype('int32')
  for samp1, samp2 in zip(y_true, y_hat):
    if np.sum(np.array(samp1)-np.array(samp2))==0:
      corr+=1
  print "overall incorrect ", len(y_true)-corr,np.round((len(y_true)-corr)/len(y_true)*100, decimals=2)
  print "overall correct ", corr, np.round(corr*100/len(y_true),decimals=2)
  return np.round(corr*100/len(y_true),decimals=2)

