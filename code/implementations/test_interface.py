from nn import NN
from torch import nn,optim
from conll_dataset import ConllWindowFromFile,sanity_window_dataset
from torch.utils.data import DataLoader
from embeddings_loader import EmbeddingsLoader
from torch.autograd import Variable
import argparse
import matplotlib.pyplot as plt
import time
import math
import numpy as np
import torch

parser = argparse.ArgumentParser(description='Collobert NN architecture applied to NER')
parser.add_argument("trained_model", help="relative path to trained model")
parser.add_argument("-d","--train_dataset", help="relative path to conll style train dataset")
parser.add_argument("-S","--sanity", help="run sanity check",action="store_true")
parser.add_argument("-G","--gpu", help="Use GPU if available.", action="store_true", default=False)
parser.add_argument("-C","--capitalization", help="Include capitalization feature",action="store_true")
parser.add_argument("-ep","--embeddings_path", help="relative path to pretrained embeddings")
parser.add_argument("-ws","--window_size", help="size of window",default=2, type=int)
parser.add_argument("-vs","--vocab_size", help="size of vocab",default=100000, type=int)
parser.add_argument("-es","--embedding_size", help="embedding dimensions",default=200, type=int)
args = parser.parse_args()

if args.sanity:
    dataset = sanity_window_dataset(args.window_size)
else:
    dataset = ConllWindowFromFile(args.train_dataset,window_size=args.window_size)

model = NN(vocab_size=args.vocab_size,number_tags=dataset.get_num_tags(),
batch_size=1,gpu=args.gpu,window_size=args.window_size,number_suffix=dataset.get_num_suffix(),
include_cap=args.capitalization)
model.load_state_dict(torch.load(args.trained_model))

if args.embeddings_path is not None:
    embLoader = EmbeddingsLoader(args.embeddings_path)
    print("Embeddings loaded, vocab size: {} embedding dims: {}".format(embLoader.get_dimensions()[0],embLoader.get_dimensions()[1]))

if args.gpu and torch.cuda.is_available():
    model.cuda()

while True:
    sentence = input("Enter sentence to process: ")
    word_list = [[w] for w in sentence.split()]
    for i in range(len(word_list)-(2*args.window_size)):
        input_tensor,_,cap_tensor,suf_tensor = model.get_train_tensors(word_list[i:i+1+2*args.window_size],word2id=embLoader.word_to_index,suffix2id=dataset.suffix_to_index)
        output = model(Variable(input_tensor),Variable(cap_tensor),Variable(suf_tensor))
        top_n, top_i = output.data.topk(1)
        tag_i = top_i[0][0]
        print("{} : {}".format(word_list[i:i+1+2*args.window_size][args.window_size],dataset.index_to_tag(tag_i)))
