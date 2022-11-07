import json
import torch
import argparse
import nltk

from nltk.corpus import wordnet as wn

from glove import GloVe

def getnode(x):
    return wn.synset_from_pos_and_offset('n', int(x[1:]))

def getwnid(u):
    s = str(u.offset())
    return 'n' + (8 - len(s)) * '0' + s

def getedges(s):
    dic = {x: i for i, x in enumerate(s)}
    edges = []
    for i, u in enumerate(s):
        for v in u.hypernyms():
            j = dict.get(v)
            if j is not None:
                edges.append((i, j))
    return edges

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/imagenet-split.json')
    args = parser.parse_args()
    
    js = json.load(open(args.input, 'r'))
    train_wnids = js['train']
    test_wnids = js['test']
    key_wnids = train_wnids + test_wnids
    
    s = list(map(getnode, key_wnids))
    wnids = list(map(getwnid, s))
    
    
    print('making glove embedding ...')
    
    glove = GloVe('data/glove.6B.300d.txt')
    vectors = []
    for wnid in wnids:
        vectors.append(glove[getnode(wnid).lemma_names()])
    vectors = torch.stack(vectors)
    
    obj = {}
    obj['wnids'] = wnids
    obj['vectors'] = vectors.tolist
    
    json.dump(obj, open('data/synset_vector.json'))