# LSTM with dropout for sequence classification in the OBS dataset
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
from sklearn.utils import shuffle

sed_val = int(sys.argv[1])
path2folds = sys.argv[2]+"/fold_"
path2models = sys.argv[3]+"/"

# fixed random seeds for reproducibility
numpy.random.seed(sed_val)
srng = RandomStreams(sed_val)

# load dataset but only keep the top n words, zero the rest
for i in range(5):
  X_train = []
  y_train = []
  for fold in range(5):
    filename = path2folds+str(fold)+"_train"
    file = open(filename,"r")
    the_dict = pickle.load(file)
    if fold==i:
      X_test = the_dict["shard"]
      y_test = the_dict["labels"]
    else:
      X_train.extend(the_dict["shard"])
      y_train.extend(the_dict["labels"])

  X_train = np.array(X_train)
  X_test = np.array(X_test)
  y_train = np.array(y_train)
  y_test = np.array(y_test)

  # load validation set
  v_path = path2folds+str(i)+"_valid"
  file = open(v_path, "r")
  the_dict = pickle.load(file)
  X_valid = np.array(the_dict["shard"])
  y_valid = np.array(the_dict["labels"])
  
  # truncate and pad input sequences
  max_sample_length = 100
  X_train = sequence.pad_sequences(X_train, maxlen=max_sample_length)
  X_test = sequence.pad_sequences(X_test, maxlen=max_sample_length)
  X_valid = sequence.pad_sequences(X_valid, maxlen=max_sample_length)
  X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
  X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
  X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], 1)
  # create the model
  embedding_vector_length = 60
  model = Sequential()
  model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2, input_shape=[max_sample_length,1]))
  model.add(Dense(4, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  print model.summary()
  n_epoch = 1
  for epoch in xrange(n_epoch):
    model.fit(X_train, y_train, nb_epoch=1, batch_size=64)
    scores = model.evaluate(X_valid, y_valid, verbose=0)
    print "Accuracy on Validation set: %.2f%%" % (scores[1]*100)
    #if scores[1]>0.9:
     # break

  #final evaluation of the model
  scores = model.evaluate(X_test, y_test, verbose=0)
  print "Accuracy on Test set: %.2f%%" % (scores[1]*100)

  # serialize model to JSON
  model_json = model.to_json()
  with open(path2models+"model_"+str(i)+".json", "w") as json_file:
    json_file.write(model_json)

  # serialize weights to HDF5
  model.save_weights(path2models+"model_"+str(i)+".h5")
  print "Saved model to disk"
