from models.chiu import Network
import torch
from torch.autograd import Variable
from loaders.conll_dataset import sanity_sequence_dataset, ConllSequenceFromFile
from loaders.embeddings_loader import EmbeddingsLoader
from torch.utils.data import DataLoader
from helper import print_tag_dict, time_since
import argparse
import os
import time
import numpy as np
import sys
import matplotlib.pyplot as plt
from loaders.batch_sampler import BatchSampler


def evaluate(output_file):
    score_file = "tmp/score_chiu"
    os.system("./conll03eval.v2 < %s > %s" % (output_file, score_file))
    print(open(score_file, "r").read())
    with open(score_file, 'r') as fin:
        fin.readline()
        line = fin.readline()
        fields = line.split(";")
        acc = float(fields[0].split(":")[1].strip()[:-1])
        precision = float(fields[1].split(":")[1].strip()[:-1])
        recall = float(fields[2].split(":")[1].strip()[:-1])
        f1 = float(fields[3].split(":")[1].strip())
    return acc, precision, recall, f1

parser = argparse.ArgumentParser(description='Chiu LSTM architecture for NER')
parser.add_argument("-d", "--train_dataset",
                    help="relative path to conll style train dataset")
parser.add_argument("-td", "--test_dataset",
                    help="relative path to conll style test dataset")
parser.add_argument("-dd", "--dev_dataset",
                    help="relative path to conll style dev dataset")
parser.add_argument("-S", "--sanity",
                    help="run sanity check", action="store_true")
parser.add_argument("-G", "--gpu",
                    help="Use GPU if available.", action="store_true", default=False)
parser.add_argument("-lr", "--learning_rate",
                    help="learning rate value", default=0.01, type=float)
parser.add_argument("-dr", "--dropout_rate",
                    help="dropout rate value", default=0.68, type=float)
parser.add_argument("-e", "--epochs",
                    help="number of epochs to train for", default=3, type=int)
parser.add_argument("-bs", "--batch_size",
                    help="batch size", default=1, type=int)
parser.add_argument("-vs", "--vocab_size",
                    help="size of vocab", default=100, type=int)
parser.add_argument("-ep", "--embeddings_path",
                    help="relative path to pretrained embeddings")
parser.add_argument("-es", "--embedding_size",
                    help="embedding dimensions", default=50, type=int)
parser.add_argument("-C", "--capitalization",
                    help="Include capitalization feature", action="store_true")
parser.add_argument("-T", "--train_emb",
                    help="Retrain embeddings", action="store_true")
parser.add_argument("-A", "--activation",
                    help="Activation function to use", choices=['relu', 'hardtanh', 'tanh'])
parser.add_argument("-hs", "--hidden_size",
                    help="LSTM hidden size", default=275, type=int)
args = parser.parse_args()

if args.sanity:
    train, test = sanity_sequence_dataset(args.vocab_size)
else:
    train = ConllSequenceFromFile(args.train_dataset,
                                  vocab_size=args.vocab_size,
                                  compute_dicts=True)
    test = ConllSequenceFromFile(args.test_dataset,
                                 vocab_size=args.vocab_size,
                                 compute_dicts=False)
    dev = ConllSequenceFromFile(args.dev_dataset,
                                 vocab_size=args.vocab_size,
                                 compute_dicts=False)
print("Train examples: {} | Test examples: {}".format(len(train), len(test)))

train_batch_sampler = BatchSampler(train, args.batch_size, enforce=args.sanity)
test_batch_sampler = BatchSampler(test, 1, enforce=args.sanity)
dev_batch_sampler = BatchSampler(dev, 1, enforce=args.sanity)
print("Train batches: {} | Test batches: {} | Batch size: {}".format(len(train_batch_sampler),
                                                                     len(test_batch_sampler),
                                                                     args.batch_size))

loader = DataLoader(train, batch_sampler=train_batch_sampler)
test_loader = DataLoader(test, batch_sampler=test_batch_sampler)
dev_loader = DataLoader(dev, batch_sampler=dev_batch_sampler)

if args.embeddings_path is not None:
    embLoader = EmbeddingsLoader(args.embeddings_path)
    print("Embeddings loaded, vocab size: {} embedding dims: {}".format(
        embLoader.get_dimensions()[0], embLoader.get_dimensions()[1]))
    args.vocab_size, args.embedding_size = embLoader.get_dimensions()


lstm = Network(args.vocab_size,
               train.get_num_tags(),
               gpu=args.gpu,
               retrain_embeddings=args.train_emb,
               embedding_dim=args.embedding_size,
               hidden_size=args.hidden_size,
               dropout=args.dropout_rate)

model_parameters = [p for p in lstm.parameters() if p.requires_grad]
params = sum([p.numel() for p in model_parameters])
print("Model with {} trainable parameters".format(params))

optimizer = torch.optim.SGD(model_parameters, lr=args.learning_rate)

if args.embeddings_path is not None:
    lstm.set_embeddings_weights(embLoader.get_weights())

if args.gpu and torch.cuda.is_available():
    lstm.cuda()

start = time.time()

# plots
train_losses = []
test_losses = []
xaxis = []

best_dev_score = 0
best_dev_epoch = 0
best_test_score = 0
best_test_epoch = 0

for epoch in range(args.epochs):
    print("Starting epoch {}.".format(epoch))
    lstm.train()
    for it, (words, tags) in enumerate(loader):
        word_tensor, tag_tensor = lstm.get_train_tensors(words,
                                                         tags,
                                                         word2id=train.word_to_index,
                                                         tag2id=train.tag_to_index,
                                                         batch_size=args.batch_size)

        optimizer.zero_grad()
        s = lstm(Variable(word_tensor))
        score = lstm.sentence_level_log_likelihood(s, tag_tensor)

        score.backward()
        # torch.nn.utils.clip_grad_norm(lstm.parameters(), 5.0)
        optimizer.step()

        if it % 1 == 0:
            print('', end='\r')
            print("Epoch {:5} Iteration {:5} Score {:7.5f}".format(epoch,
                                                                   it,
                                                                   score.data[0]),
                  end='')
    print('\n')
    # eval dev dataset
    lstm.eval()
    tagged_file = open("tmp/dev.annotated", "w")
    for words, tags in dev_loader:
        word_tensor, tag_tensor = lstm.get_train_tensors(words,
                                                         tags,
                                                         word2id=train.word_to_index,
                                                         tag2id=train.tag_to_index,
                                                         batch_size=1)
        s = lstm(Variable(word_tensor))
        score = lstm.sentence_level_log_likelihood(s, tag_tensor)

        for batch in range(1):
            _, tag_list = lstm.viterbi_decode(s[batch])
            for i, tag_i in enumerate(tag_list):
                pred_tag = train.index_to_tag(tag_i)
                real_tag = tags[i][batch]
                word = words[i][batch]
                tagged_file.write("{} {} {}\n".format(word, real_tag, pred_tag))
            tagged_file.write("\n")
    tagged_file.close()
    acc, prec, rec, f1 = evaluate("tmp/dev.annotated")
    if f1 > best_dev_score:
        best_dev_score = f1
        best_dev_epoch = epoch
        # evaluate test set
        tagged_file = open("tmp/test.annotated", "w")
        for words, tags in test_loader:
            word_tensor, tag_tensor = lstm.get_train_tensors(words,
                                                             tags,
                                                             word2id=train.word_to_index,
                                                             tag2id=train.tag_to_index,
                                                             batch_size=1)
            s = lstm(Variable(word_tensor))
            score = lstm.sentence_level_log_likelihood(s, tag_tensor)

            for batch in range(1):
                _, tag_list = lstm.viterbi_decode(s[batch])
                for i, tag_i in enumerate(tag_list):
                    pred_tag = train.index_to_tag(tag_i)
                    real_tag = tags[i][batch]
                    word = words[i][batch]
                    tagged_file.write("{} {} {}\n".format(word, real_tag, pred_tag))
                tagged_file.write("\n")
        tagged_file.close()
        acc, prec, rec, f1 = evaluate("tmp/test.annotated")
        if f1 > best_test_score:
            best_test_score = f1
            best_test_epoch = epoch

    print("\nBest dev F1 score {} at epoch {}".format(best_dev_score, best_dev_epoch))
    print("Best test F1 score {} at epoch {}".format(best_test_score, best_test_epoch))
