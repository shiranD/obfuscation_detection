from keras.models import model_from_json
from keras.preprocessing import sequence
import pickle
import numpy as np
import sys
from aux_func import acc, overall_acc, align_w_out, concatit
import re
import keras.backend as K

def my_mse(y_true, y_pred):
  return K.mean(K.sum(K.square(y_true-y_pred),axis=1), axis=0)

def flat(vec):
  return [item for sublist in vec for item in sublist]

num_f = 5
max_sample_length = 100
y_ref_all = []
y_hat_all = []
pattern = r"_\d+"
path2folds = sys.argv[1]
path2models = sys.argv[2]

for fold in xrange(num_f):
  m_path0 = path2models + "/model_0"+str(fold)+".json"
  w_path0 = path2models + "/model_0"+str(fold)+".h5"
  m_path1 = path2models + "/model_1"+str(fold)+".json"
  w_path1 = path2models + "/model_1"+str(fold)+".h5"
  m_path2 = path2models + "/model_2"+str(fold)+".json"
  w_path2 = path2models + "/model_2"+str(fold)+".h5"
  m_path3 = path2models + "/model_3"+str(fold)+".json"
  w_path3 = path2models + "/model_3"+str(fold)+".h5"
  m_path4 = path2models + "/model_4"+str(fold)+".json"
  w_path4 = path2models + "/model_4"+str(fold)+".h5"
  m_path5 = path2models + "/model_5"+str(fold)+".json"
  w_path5 = path2models + "/model_5"+str(fold)+".h5"
  m_path6 = path2models + "/model_6"+str(fold)+".json"
  w_path6 = path2models + "/model_6"+str(fold)+".h5"
  m_path7 = path2models + "/model_7"+str(fold)+".json"
  w_path7 = path2models + "/model_7"+str(fold)+".h5"

  # load model
  json_file = open(m_path0, "r")
  loaded_model_json = json_file.read()
  json_file.close()
  model0 = model_from_json(loaded_model_json)
  # load weights
  model0.load_weights(w_path0)
  model0.compile(loss=my_mse, optimizer='adam', metrics=['accuracy'])

  json_file = open(m_path1, "r")
  loaded_model_json = json_file.read()
  json_file.close()
  model1 = model_from_json(loaded_model_json)
  # load weights
  model1.load_weights(w_path1)
  model1.compile(loss=my_mse, optimizer='adam', metrics=['accuracy'])

  json_file = open(m_path2, "r")
  loaded_model_json = json_file.read()
  json_file.close()
  model2 = model_from_json(loaded_model_json)
  # load weights
  model2.load_weights(w_path2)
  model2.compile(loss=my_mse, optimizer='adam', metrics=['accuracy'])

  json_file = open(m_path3, "r")
  loaded_model_json = json_file.read()
  json_file.close()
  model3 = model_from_json(loaded_model_json)
  # load weights
  model3.load_weights(w_path3)
  model3.compile(loss=my_mse, optimizer='adam', metrics=['accuracy'])

  json_file = open(m_path4, "r")
  loaded_model_json = json_file.read()
  json_file.close()
  model4 = model_from_json(loaded_model_json)
  # load weights
  model4.load_weights(w_path4)
  model4.compile(loss=my_mse, optimizer='adam', metrics=['accuracy'])

  json_file = open(m_path5, "r")
  loaded_model_json = json_file.read()
  json_file.close()
  model5 = model_from_json(loaded_model_json)
  # load weights
  model5.load_weights(w_path5)
  model5.compile(loss=my_mse, optimizer='adam', metrics=['accuracy'])

  json_file = open(m_path6, "r")
  loaded_model_json = json_file.read()
  json_file.close()
  model6 = model_from_json(loaded_model_json)
  # load weights
  model6.load_weights(w_path6)
  model6.compile(loss=my_mse, optimizer='adam', metrics=['accuracy'])

  json_file = open(m_path7, "r")
  loaded_model_json = json_file.read()
  json_file.close()
  model7 = model_from_json(loaded_model_json)
  # load weights
  model7.load_weights(w_path7)
  model7.compile(loss=my_mse, optimizer='adam', metrics=['accuracy'])

  # load X_test and y_test
  fold_file = path2folds + "/fold_"+str(fold)+"_train"
  file = open(fold_file, "r")
  the_dict = pickle.load(file)
  X_test = np.array(the_dict["shard"])
  y_ref = np.array(the_dict["labels"])
  # extract predictions
  y_hat0 = model0.predict(X_test, verbose=0)
  y_hat0 = flat(y_hat0)
  inputL = concatit(X_test, y_hat0)
  y_hat1 = model1.predict(inputL, verbose=0)
  y_hat1 = flat(y_hat1)
  inputL = concatit(X_test, y_hat1)
  y_hat2 = model2.predict(inputL, verbose=0)
  y_hat2 = flat(y_hat2)
  inputL = concatit(X_test, y_hat2)
  y_hat3 = model3.predict(inputL, verbose=0)
  y_hat3 = flat(y_hat3)
  inputL = concatit(X_test, y_hat3)
  y_hat4 = model4.predict(inputL, verbose=0)
  y_hat4 = flat(y_hat4)
  inputL = concatit(X_test, y_hat4)
  y_hat5 = model5.predict(inputL, verbose=0)
  y_hat5 = flat(y_hat5)
  inputL = concatit(X_test, y_hat5)
  y_hat6 = model6.predict(inputL, verbose=0)
  y_hat6 = flat(y_hat6)
  inputL = concatit(X_test, y_hat6)
  y_hat7 = model7.predict(inputL, verbose=0)
  y_hat7 = flat(y_hat7)

  y_hat = align_w_out(y_hat0, y_hat1, y_hat2, y_hat3, y_hat4, y_hat5, y_hat6, y_hat7)
  y_hat = (y_hat>0.5).astype('int32')
  y_ref = y_ref.astype('int32')

  if 0: 
    # score for bit level
    acc(y_ref[:,0], y_hat0)
    acc(y_ref[:,1], y_hat1)
    acc(y_ref[:,2], y_hat2)
    acc(y_ref[:,3], y_hat3)
    acc(y_ref[:,4], y_hat4)
    acc(y_ref[:,5], y_hat5)
    acc(y_ref[:,6], y_hat6)
    acc(y_ref[:,7], y_hat7)
    # score for sample level
    overall_acc(y_ref, y_hat)
  if 1:
    for (ref, hat) in zip(y_ref, y_hat):
    # compute key
      key = 0
      for cnt,s in enumerate(ref[::-1]):
        key+=s*2**cnt
      for m, (r, h) in enumerate(zip(ref,hat)):
        if r == h:
          print m, key, "yes"
        else:
          print m, key, "no"

  if 0:
    for (ref, hat) in zip(y_ref, y_hat):
      # compute key
      key = 0
      for cnt,s in enumerate(ref[::-1]):
        key+=s*2**cnt
        # compare
      ref = np.array(ref)
      hat = np.array(hat)

      #print ref
      #print hat
      flag=0
      for (y1,y2) in zip(ref,hat):
        if y1!=y2:
          flag = 1
          break
      if flag:
        print key, "no"
      else:
        print key, "yes"
  if 0:
    # length vs acc
    for (ref, hat, sample) in zip(y_ref, y_hat, X_test):
      idx = next((i for i, x in enumerate(sample[::-1]) if np.sum(x)), None)
      idx = len(sample)-idx
      summ = np.sum(ref-hat)
      if summ:
        print idx, "no"
      else:
        print idx, "yes"
  if 0: 
    # length vs bit acc
    for (ref, hat, sample) in zip(y_ref, y_hat, X_test):
      idx = next((i for i, x in enumerate(sample[::-1]) if np.sum(x)), None)
      idx = len(sample)-idx
      for g, (yh, yt) in enumerate(zip(hat, ref)):
        if yh==yt:
      	  print idx, g, "same"
        else:
	  print idx, g, "not"
