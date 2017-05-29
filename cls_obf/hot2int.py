import numpy as np

def hot2int(samples):
  "convert from one hot rep to int rep"
  new = []
  for vec in samples:
    grp = np.argmax(vec)
    new.append(grp)
  return new
