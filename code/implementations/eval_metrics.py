from nn import NN
from torch import nn,optim
from conll_dataset import ConllWindowFromFile
from torch.utils.data import DataLoader
from embeddings_loader import EmbeddingsLoader
from torch.autograd import Variable
from helper import loadDatasets
import argparse
import matplotlib.pyplot as plt
import time
import math
import numpy as np
import torch

parser = argparse.ArgumentParser(description='Collobert NN architecture applied to NER')
parser.add_argument("trained_model", help="relative path to trained model")
parser.add_argument("-d","--train_dataset", help="relative path to conll style train dataset")
parser.add_argument("-td","--test_dataset", help="relative path to conll style test dataset")
parser.add_argument("-bs","--batch_size", help="batch size",default=1, type=int)
parser.add_argument("-G","--gpu", help="Use GPU if available.", action="store_true", default=False)
parser.add_argument("-ep","--embeddings_path", help="relative path to pretrained embeddings")
parser.add_argument("-ws","--window_size", help="size of window",default=2, type=int)
args = parser.parse_args()

dataset,loader,test_dataset,test_loader = loadDatasets(args.train_dataset,
args.test_dataset,args.window_size,args.batch_size,shuffle=False)
print("Number of training examples / test examples: {} / {}".format(len(dataset),len(test_dataset)))

if args.embeddings_path is not None:
    embLoader = EmbeddingsLoader(args.embeddings_path)
    print("Embeddings loaded, vocab size: {} embedding dims: {}".format(embLoader.get_dimensions()[0],embLoader.get_dimensions()[1]))
    args.vocab_size,args.embedding_size = embLoader.get_dimensions()

model = NN(vocab_size=args.vocab_size,number_tags=dataset.get_num_tags(),batch_size=args.batch_size,gpu=args.gpu,window_size=args.window_size)
model.load_state_dict(torch.load(args.trained_model))

if args.gpu and torch.cuda.is_available():
    model.cuda()

tag_level = {
'correct':0,
'total_in_test': 0,
'total_detected': 0
}

for i,(words_input,tags) in enumerate(test_loader):
    input_tensor,target_tensor = model.get_train_tensors(words_input,tags,embLoader.word_to_index,dataset.tag_to_index)
    output = model(Variable(input_tensor))
    top_n, top_i = output.data.topk(1)
    for i,tag_i in enumerate(top_i):
        tag_i = tag_i[0]
        if tag_i == target_tensor[i] and target_tensor[i] != dataset.tag_to_index('O'):
            tag_level['correct'] += 1
            tag_level['total_detected'] += 1
        elif tag_i != dataset.tag_to_index('O'):
            tag_level['total_detected'] += 1
        if target_tensor[i] != dataset.tag_to_index('O'):
            tag_level['total_in_test'] += 1

tag_level['precision'] = tag_level['correct']/tag_level['total_detected']
tag_level['recall'] = tag_level['correct']/tag_level['total_in_test']
tag_level['F1'] = 2*(tag_level['precision']*tag_level['recall'])/(tag_level['precision']+tag_level['recall'])

""""
    Precision = Total correct NEs detected / Total detected
    Recall = Total correct NEs detected / Total NEs in test set
    F1 = 2*(precision*recall)/(precision+recall)
"""
print("\nEvaluation Metrics")
print(" --- TAG LEVEL --- ")
for key,val in tag_level.items():
    print("{:20} | {:4.3f}".format(key,val))
print(" --- NE LEVEL --- ")
