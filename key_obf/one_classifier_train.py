from __future__ import division
import theano
import numpy
import numpy as np
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from theano.tensor.shared_randomstreams import RandomStreams
import sys
import pickle
import keras.backend as K
from aux_func1 import acc
import pdb
from random import randint

def multilabel_crossentropy(y_true, y_pred):
  return -K.mean(K.sum(K.log(y_pred) * y_true + K.log(1 - y_pred) * (1 - y_true), axis=1), axis=0)

sed_val = int(sys.argv[1])
path2folds = sys.argv[2]+"/fold_"
path2models = sys.argv[3]+"/"

# fix random seeds for reproducibility
numpy.random.seed(sed_val)
srng = RandomStreams(sed_val)
# load dataset but only keep the top n words, zero the rest
max_sample_length = 100

for i in range(5):
  model = Sequential()
  model.add(LSTM(100,input_shape=[max_sample_length,8]))
  model.add(Dense(8, activation='sigmoid'))
  model.compile(loss=multilabel_crossentropy, optimizer='adam', metrics=['accuracy'])
  print model.summary()

  X_train = []
  y_train = []
  filename = path2folds+str(i)+"_valid"
  file = open(filename,"r")
  the_dict = pickle.load(file)
  X_valid = the_dict["shard"]
  y_valid = the_dict["labels"]
  X_valid = np.array(X_valid)
  y_valid = np.array(y_valid)

  for fold in range(5):
    # load folds
    filename = path2folds+str(fold)+"_train"
    file = open(filename,"r")
    the_dict = pickle.load(file)
    if fold==i:
      X_test = the_dict["shard"]
      y_test = the_dict["labels"]
      continue
    if X_train == []:
      X_train = np.array(the_dict["shard"])
      y_train = the_dict["labels"]
    else:
      X_train = np.concatenate((X_train, np.array(the_dict["shard"])), axis=0)
      y_train.extend(the_dict["labels"])

  X_test = np.array(X_test)
  y_train = np.array(y_train)
  y_test = np.array(y_test)
  # train
  k =  int(len(y_train)/20)
  for ep in xrange(20):
    print "round ", ep
    model.fit(X_train[ep*k:k*(ep+1)], y_train[k*ep:k*(ep+1)], nb_epoch=1, batch_size=1)
    y_hat = model.predict(X_train[ep*k:k*(ep+1)])
    co = acc(y_train[k*ep:k*(ep+1)], y_hat)
    y_hat = model.predict(X_valid)
    co = acc(y_valid, y_hat)
    if co > 30:
      break

  # predict on test
  y_hat = model.predict(X_test)
  print "Test Accuracy"
  co = acc(y_test, y_hat)

  # serialize model to JSON
  model_json = model.to_json()
  with open(path2models+"/model_"+str(i)+".json", "w") as json_file:
    json_file.write(model_json)

  # serialize weights to HDF5
  model.save_weights(path2models+"/model_"+str(i)+".h5")
  print "Saved model to disk"
