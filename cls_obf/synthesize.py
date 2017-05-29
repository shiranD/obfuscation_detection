import codecs
import copy
import gzip #Compression compatible with gzip
from random import randint, seed
import sys

sed_val = int(sys.argv[1])
data_path = sys.argv[2]
seed(sed_val)

"""the ourpose of this script is to generate various obfuscated hex-writen samples"""

def obfuscate_base64(s):
    so = codecs.encode(s,"base64")
    lenn = randint(10,len(s))
    pos = randint(0,len(so)-lenn)
    so = so[pos:lenn+pos]
    so = ''.join( [ "%2x " %x for x in so ] )
    return so

def obfuscate_xor(s, k):
    lenn = randint(10,len(s))
    pos = randint(0,len(s)-lenn)
    s = s[pos:lenn+pos]
    so = "".join([ "%2x " %(b^k) for b in s])
    return so

def obfuscate_gz(s):
    so = gzip.compress(s)[10:]
    lenn = randint(10,len(so))
    pos = randint(0,min(len(so)-lenn,100))
    so = so[pos:lenn+pos]
    so = ''.join( [ "%2x " %x  for x in so ] )
    return so

nsamples = 100000
max_obfuscation_depth = 1

obfuscation_types = [obfuscate_base64, obfuscate_xor, obfuscate_gz]
obfuscation_types_string = ['base64', 'xor', 'gz']

f = open(data_path,"rb")

for n in range(nsamples):
    b2t_label = ""
    ln = randint(10,80)
    loc = randint(0,10000000)
    f.seek(loc)
    text = f.read(ln)
    text_obfuscated = copy.copy(text)
    obfuscation_type_idx = randint(0, len(obfuscation_types))

    if obfuscation_type_idx == 0:
        text_obfuscated = ''.join( [ "%2x " %x  for x in text_obfuscated ] )
        label = 'plaintext'
        b2t_label+=label+" 1"
    else:
        obfuscation_type_idx-=1        
        obfuscation_type = obfuscation_types[obfuscation_type_idx]
        label = obfuscation_types_string[obfuscation_type_idx]

        if label == 'xor':
            key = randint(1,255)
            text_obfuscated = obfuscation_type(text_obfuscated,key)
            label += '_' + str(key)
        else:
            text_obfuscated = obfuscation_type(text_obfuscated)

        b2t_label+=label+" 1"
    text_obfuscated+=" "+b2t_label
    print(text_obfuscated)
