# LSTM with dropout for sequence classification in the OBS dataset
from __future__ import division
import theano
import numpy
import numpy as np
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing import sequence
from theano.tensor.shared_randomstreams import RandomStreams
import sys
import pickle
import keras.backend as K
from aux_func import align_w_out, concatit, acc, overall_acc

def my_mse(y_true, y_pred):
  return K.mean(K.sum(K.square(y_true-y_pred),axis=1), axis=0)

def flat(vec):
   return [item for sublist in vec for item in sublist]

# load dataset but only keep the top n words, zero the rest
sed_val = int(sys.argv[1])
path2folds = sys.argv[2]+"/fold_"
path2models = sys.argv[3]

max_sample_length = 100
numpy.random.seed(sed_val)
srng = RandomStreams(sed_val)

def bit_model(num):
  model = Sequential()
  inp_ln = max_sample_length+num
  model.add(LSTM(100, input_shape=[inp_ln, 8]))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss=my_mse, optimizer='adam', metrics=['accuracy'])
  print model.summary()
  return model

for i in [4]:

  model_0 = bit_model(0)
  model_1 = bit_model(1)
  model_2 = bit_model(1)
  model_3 = bit_model(1)
  model_4 = bit_model(1)
  model_5 = bit_model(1)
  model_6 = bit_model(1)
  model_7 = bit_model(1)

  for fold in range(5):
    X_train = np.array([])
    y_train = []
    filename = path2folds+str(fold)+"_train"
    file = open(filename,"r")
    the_dict = pickle.load(file)
    if fold==i:
      X_test = np.array(the_dict["shard"])
      y_test = np.array(the_dict["labels"])
      X_test = sequence.pad_sequences(X_test, maxlen=max_sample_length)
      X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 8)
      continue
    else:
      X_train = np.array(the_dict["shard"])
      y_train = np.array(the_dict["labels"])
      X_train = sequence.pad_sequences(X_train, maxlen=max_sample_length)
      #X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 8)
      k = int(len(y_train)/5)
      #print X_train.shape[0], X_train.shape[1], X_train.shape[2]
      print "fold "+str(fold)
      for ep in xrange(5):
        print "round ", ep
        print "1st bit"
        x = X_train[ep*k:k*(ep+1)]
        model_0.fit(x, y_train[k*ep:k*(ep+1),0], nb_epoch=1, batch_size=1)
        y_hat = model_0.predict(X_train[ep*k:k*(ep+1)])
        y_hat0 = flat(y_hat)
        acc(y_train[k*ep:k*(ep+1),0], y_hat0)
        print "2nd bit"
        # concatenate x with y
        inputL = concatit(x, y_hat0)
        model_1.fit(inputL, y_train[k*ep:k*(ep+1),1], nb_epoch=1, batch_size=1)
        y_hat = model_1.predict(inputL)
        y_hat1 = flat(y_hat)
        acc(y_train[k*ep:k*(ep+1),1], y_hat1)
        print "3rd bit"
        inputL = concatit(x, y_hat1)
        model_2.fit(inputL, y_train[k*ep:k*(ep+1),2], nb_epoch=1, batch_size=1)
        y_hat = model_2.predict([inputL])
        y_hat2 = flat(y_hat)
        acc(y_train[k*ep:k*(ep+1),2], y_hat2)
        print "4th bit"
        inputL = concatit(x, y_hat2)
        model_3.fit(inputL, y_train[k*ep:k*(ep+1),3], nb_epoch=1, batch_size=1)
        y_hat = model_3.predict(inputL)
        y_hat3 = flat(y_hat)
        acc(y_train[k*ep:k*(ep+1),3], y_hat3)
        print "5th bit"
        inputL = concatit(x, y_hat3)
        model_4.fit(inputL, y_train[k*ep:k*(ep+1),4], nb_epoch=1, batch_size=1)
        y_hat = model_4.predict(inputL)
        y_hat4 = flat(y_hat)
        acc(y_train[k*ep:k*(ep+1),4], y_hat4)
        print "6th bit"
        inputL = concatit(x, y_hat4)
        model_5.fit(inputL, y_train[k*ep:k*(ep+1),5], nb_epoch=1, batch_size=1)
        y_hat = model_5.predict(inputL)
        y_hat5 = flat(y_hat)
        acc(y_train[k*ep:k*(ep+1),5], y_hat5)
        print "7th bit"
        inputL = concatit(x, y_hat5)
        model_6.fit(inputL, y_train[k*ep:k*(ep+1),6], nb_epoch=1, batch_size=1)
        y_hat = model_6.predict(inputL)
        y_hat6 = flat(y_hat)
        acc(y_train[k*ep:k*(ep+1),6], y_hat6)
        print "8th bit"
        inputL = concatit(x, y_hat6)
        model_7.fit(inputL, y_train[k*ep:k*(ep+1),7], nb_epoch=1, batch_size=1)
        y_hat = model_7.predict(inputL)
        y_hat7 = flat(y_hat)
        acc(y_train[k*ep:k*(ep+1),7], y_hat7)

        
        y_hat = align_w_out(y_hat0, y_hat1, y_hat2, y_hat3, y_hat4, y_hat5, y_hat6, y_hat7)
        ac = overall_acc(np.array(y_train[ep*k:k*(ep+1)]), y_hat)
        if ac > 30:
	  break
  print "Test Accuracy"
  y_t0 = model_0.predict(X_test)
  y_t0 = flat(y_t0)
  inputL = concatit(X_test, y_t0)
  y_t1 = model_1.predict(inputL)
  y_t1 = flat(y_t1)
  inputL = concatit(X_test, y_t1)
  y_t2 = model_2.predict(inputL)
  y_t2 = flat(y_t2)
  inputL = concatit(X_test, y_t2)
  y_t3 = model_3.predict(inputL)
  y_t3 = flat(y_t3)
  inputL = concatit(X_test, y_t3)
  y_t4 = model_4.predict(inputL)
  y_t4 = flat(y_t4)
  inputL = concatit(X_test, y_t4)
  y_t5 = model_5.predict(inputL)
  y_t5 = flat(y_t5)
  inputL = concatit(X_test, y_t5)
  y_t6 = model_6.predict(inputL)
  y_t6 = flat(y_t6)
  inputL = concatit(X_test, y_t6)
  y_t7 = model_7.predict(inputL)
  y_t7 = flat(y_t7)
  y_hat =  align_w_out(y_t0, y_t1, y_t2, y_t3, y_t4, y_t5, y_t6, y_t7)
  ac = overall_acc(np.array(y_test), y_hat)
  # serialize model to JSON
  model_json = model_0.to_json()
  with open(path2models+"/model_0"+str(i)+".json", "w") as json_file:
    json_file.write(model_json)
  # serialize weights to HDF5
  model_0.save_weights(path2models+"/model_0"+str(i)+".h5")
  
  model_json = model_1.to_json()
  with open(path2models+"/model_1"+str(i)+".json", "w") as json_file:
    json_file.write(model_json)
  # serialize weights to HDF5
  model_1.save_weights(path2models+"/model_1"+str(i)+".h5")
  
  model_json = model_2.to_json()
  with open(path2models+"/model_2"+str(i)+".json", "w") as json_file:
    json_file.write(model_json)
  # serialize weights to HDF5
  model_2.save_weights(path2models+"/model_2"+str(i)+".h5")
  
  model_json = model_3.to_json()
  with open(path2models+"/model_3"+str(i)+".json", "w") as json_file:
    json_file.write(model_json)
  # serialize weights to HDF5
  model_3.save_weights(path2models+"/model_3"+str(i)+".h5")
  
  model_json = model_4.to_json()  
  with open(path2models+"/model_4"+str(i)+".json", "w") as json_file:
    json_file.write(model_json)
  # serialize weights to HDF5
  model_4.save_weights(path2models+"/model_4"+str(i)+".h5")
  
  model_json = model_5.to_json()
  with open(path2models+"/model_5"+str(i)+".json", "w") as json_file:
    json_file.write(model_json)
  # serialize weights to HDF5
  model_5.save_weights(path2models+"/model_5"+str(i)+".h5")
  
  model_json = model_6.to_json()  
  with open(path2models+"/model_6"+str(i)+".json", "w") as json_file:
    json_file.write(model_json)
  # serialize weights to HDF5
  model_6.save_weights(path2models+"/model_6"+str(i)+".h5")
  
  model_json = model_7.to_json()
  with open(path2models+"/model_7"+str(i)+".json", "w") as json_file:
    json_file.write(model_json)
  # serialize weights to HDF5
  model_7.save_weights(path2models+"/model_7"+str(i)+".h5")
  
  print "Saved model to disk"
