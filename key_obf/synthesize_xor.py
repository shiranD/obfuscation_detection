import copy
from random import randint, seed, uniform
import sys

sed_val = int(sys.argv[1])
data_path = sys.argv[2]
seed(sed_val)

"""the ourpose of this script is to generate various XOR keys obfuscated hex-writen samples"""

def obfuscate_xor(s, k):
    #so = [hex(b^k)[2:] for b in s]
    so = "".join([ "%2x " %(b^k) for b in s])
    return so

nsamples = 100000
max_obfuscation_depth = 1

obfuscation_types = [obfuscate_xor]
obfuscation_types_string = ['xor']

f = open(data_path,"rb")

for n in range(nsamples):
    # labels to anticipate
    b2t_label = ""
    ln = randint(10,80)
    loc = randint(0,10000000)
    f.seek(loc)
    text = f.read(ln)
    text_obfuscated = copy.copy(text)
    obfuscation_type_idx=0        
    obfuscation_type = obfuscation_types[obfuscation_type_idx]
    label = obfuscation_types_string[obfuscation_type_idx]

    if label == 'xor':
    	#print text_obfuscated
        key = randint(1,255)
        text_obfuscated = obfuscation_type(text_obfuscated,key)
        label += '_' + str(key)
    else:
    	text_obfuscated = obfuscation_type(text_obfuscated)

    b2t_label+=label+" 1"
    text_obfuscated+=" "+b2t_label
    print(text_obfuscated)
