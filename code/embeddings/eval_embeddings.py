import argparse
from scipy import spatial
import numpy as np
import io
from random import shuffle

parser = argparse.ArgumentParser(description='Analyze embedding performance on analogies')
parser.add_argument("file", help="relative path to embedding file")
parser.add_argument("-a","--analogies", help="relative path to analogies file",default="./i_analogies")
args = parser.parse_args()

word2emb = dict()

#load embeddings
print("Loading embeddings")
with io.open(args.file,mode="r",encoding="utf-8") as f:
    for line in f:
        s = line.split(" ")
        if len(s) <= 2 :
            continue
        word2emb[s[0]] = [float(x) for x in s[1:]]

#load analogies
print("Loading analogies")
with io.open(args.analogies,mode="r",encoding="utf-8") as f:
    analog = f.read()
    analog = analog.split("\n")
    shuffle(analog)
    for line in analog:
        s = line.split()
        if s[0] == ':':
            continue
        s = [w.lower() for w in s]
        embeddings = []
        skip_example = False
        for w in s:
            if w not in word2emb:
                skip_example = True
                break
            embeddings.append(np.array(word2emb[w]))
        if skip_example:
            continue
        #analogies: w1 - w2 + w4 = w3
        prediction = embeddings[0]-embeddings[1]+embeddings[3]
        k = 3
        biggest_dist_key = ' '
        top_k = dict()
        top_k[biggest_dist_key] = float("inf")
        for w,emb in word2emb.items():
            if w in [s[0],s[1],s[3]]: continue
            d = spatial.distance.cosine(prediction, np.array(emb))
            #update top-k dict
            if d < top_k[biggest_dist_key]:
                if len(top_k) >= k:
                    del top_k[biggest_dist_key]
                    biggest_dist_key = sorted(top_k.items(),key=lambda t: t[1])[-1][0]
                top_k[w] = d
            else:
                if len(top_k) < k:
                    biggest_dist_key = w
                    top_k[w] = d
        print(u'{}\n{} - {} + {} = {}.'.format("-"*25,s[0],s[1],s[3],s[2]))
        for m,d in sorted(top_k.items(),key=lambda t: t[1]):
            print(u'{} {}'.format(m,d))
