import json

from nltk.corpus import wordnet as wn
import torch
from glove import Glove

def getnod(x):
    return wn.synset_from_pos_and_offset('n', int(x[1:]))     # query synset by id

def getwnid(u):
    s = str(u.offset())
    return 'n' + (8 - len(s)) * '0' + s

def getedges(s):
    dic = {x: i for i, x in enumerate(s)}
    edges = []
    for i, u in enumerate(s):
        for v in u.hypernyms():
            