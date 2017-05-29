from sklearn.metrics import accuracy_score, confusion_matrix 
from keras.models import model_from_json
from keras.preprocessing import sequence
import pickle
import numpy as np
import sys
from hot2int import hot2int
from collections import defaultdict
import re 
import sys

path2folds = sys.argv[1]+"/"
path2models = sys.argv[2]+"/model_"

num_f = 5
max_sample_length = 100
pattern = r"_\d+"

for fold in xrange(num_f):
  m_path = path2models+str(fold)+".json"
  w_path = path2models+str(fold)+".h5"

  # load model
  json_file = open(m_path, "r")
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)

  # load weights
  loaded_model.load_weights(w_path)
  loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  # load X_test and y_test
  fold_file = path2folds+"fold_"+str(fold)+"_train"
  file = open(fold_file, "r")
  the_dict = pickle.load(file)
  X_test = np.array(the_dict["shard"])
  y_ref = np.array(the_dict["labels"])
  X_test = sequence.pad_sequences(X_test, maxlen=max_sample_length) 
  X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

  # extract predictions
  y_hat = loaded_model.predict(X_test, verbose=0)

  # convert ref and pred to int class
  y_hat = hot2int(y_hat)
  y_ref = hot2int(y_ref)
  ac_score = accuracy_score(y_ref, y_hat)
  print "Overall Accuracy is:", ac_score*100
  print "Confusion Matrix"
  # display the small confusion matrix
  print '{:8} {:6} {:6} {:6} {:6}'.format("grp ","gz", "base64", "text", "xor")
  obs = ["gz", "base64", "text", "xor"]
  conf = confusion_matrix(y_ref,y_hat)
  for i,line in enumerate(conf):
    print '{:8} {}'.format(obs[i], line)

