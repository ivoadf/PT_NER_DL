from models.collobert import NN
from torch import nn, optim
from loaders.conll_dataset import sanity_window_dataset, ConllWindowFromFile
from helper import time_since, print_tag_dict, evaluate
from loaders.batch_sampler import BatchSampler
from torch.utils.data import DataLoader
from loaders.embeddings_loader import EmbeddingsLoader
from torch.autograd import Variable
from os import path
import argparse
import matplotlib.pyplot as plt
import time
import numpy as np
import torch
import sys

parser = argparse.ArgumentParser(description='Collobert NN architecture applied to NER')
parser.add_argument("-d", "--train_dataset",
                    help="relative path to conll style train dataset")
parser.add_argument("-td", "--test_dataset",
                    help="relative path to conll style test dataset")
parser.add_argument("-dd", "--dev_dataset",
                    help="relative path to conll style dev dataset")
parser.add_argument("-S", "--sanity",
                    help="run sanity check", action="store_true")
parser.add_argument("-C", "--capitalization",
                    help="Include capitalization feature", action="store_true")
parser.add_argument("-T", "--train_emb",
                    help="Retrain embeddings", action="store_true")
parser.add_argument("-A", "--activation", help="Activation function to use",
                    choices=['relu', 'hardtanh', 'tanh'], default='hardtanh')
parser.add_argument("-G", "--gpu",
                    help="Use GPU if available.", action="store_true", default=False)
parser.add_argument("-lr", "--learning_rate",
                    help="learning rate value", default=0.0075, type=float)
parser.add_argument("-e", "--epochs",
                    help="number of epochs to train for", default=3, type=int)
parser.add_argument("-ep", "--embeddings_path",
                    help="relative path to pretrained embeddings")
parser.add_argument("-bs", "--batch_size",
                    help="batch size", default=1, type=int)
parser.add_argument("-ws", "--window_size",
                    help="size of window", default=2, type=int)
parser.add_argument("-vs", "--vocab_size",
                    help="size of vocab", default=100000, type=int)
parser.add_argument("-es", "--embedding_size",
                    help="embedding dimensions", default=200, type=int)
args = parser.parse_args()

if args.sanity:
    dataset, test_dataset = sanity_window_dataset(args.window_size, args.vocab_size)
    print("SANITY CHECK: {} training examples".format(len(dataset)))
else:
    dataset = ConllWindowFromFile(args.train_dataset, args.window_size,
                                  compute_dicts=True, vocab_size=args.vocab_size)
    test_dataset = ConllWindowFromFile(args.test_dataset, args.window_size,
                                       compute_dicts=False, vocab_size=args.vocab_size)
    dev_dataset = ConllWindowFromFile(args.dev_dataset, args.window_size,
                                       compute_dicts=False, vocab_size=args.vocab_size)
    print("Number of training examples / test examples: {} / {}".format(len(dataset),
                                                                        len(test_dataset)))

train_batch_sampler = BatchSampler(dataset, args.batch_size, enforce=args.sanity)
test_batch_sampler = BatchSampler(test_dataset, 1, enforce=args.sanity)
dev_batch_sampler = BatchSampler(dev_dataset, 1, enforce=args.sanity)
print("Train batches: {} | Test batches: {} | Batch size: {}".format(len(train_batch_sampler),
                                                                     len(test_batch_sampler),
                                                                     args.batch_size))
loader = DataLoader(dataset, batch_sampler=train_batch_sampler)
test_loader = DataLoader(test_dataset, batch_sampler=test_batch_sampler)
dev_loader = DataLoader(dev_dataset, batch_sampler=dev_batch_sampler)

if args.embeddings_path is not None:
    embLoader = EmbeddingsLoader(args.embeddings_path)
    print("Embeddings loaded, vocab size: {} embedding dims: {}".format(
        embLoader.get_dimensions()[0], embLoader.get_dimensions()[1]))
    args.vocab_size, args.embedding_size = embLoader.get_dimensions()

model = NN(vocab_size=args.vocab_size, number_tags=dataset.get_num_tags(),
           batch_size=args.batch_size, gpu=args.gpu, window_size=args.window_size,
           embedding_dim=args.embedding_size, retrain_embeddings=args.train_emb,
           include_cap=args.capitalization, activation=args.activation,
           number_suffix=dataset.get_num_suffix())

model_parameters = [p for p in model.parameters() if p.requires_grad]
params = sum([p.numel() for p in model_parameters])
print("Model with {} trainable parameters".format(params))

if args.embeddings_path is not None:
    model.set_embeddings_weights(embLoader.get_weights())

if args.gpu and torch.cuda.is_available():
    model.cuda()

optimizer = optim.SGD(model_parameters, lr=args.learning_rate)

dev_best_f1 = 0
dev_best_epoch = 0
test_best_f1 = 0
test_best_epoch = 0

start = time.time()

if args.embeddings_path is not None:
    word_to_index = embLoader.word_to_index
else:
    word_to_index = dataset.word_to_index

if args.gpu and torch.cuda.is_available():
    v_type = torch.cuda.FloatTensor
else:
    v_type = torch.FloatTensor

for epoch in range(args.epochs):
    print("Starting epoch {}.".format(epoch))
    model.set_batch_size(args.batch_size)
    for i, (words_input, tags) in enumerate(loader):
        input_tensor, target_tensor, cap_tensor, suf_tensor = \
            model.get_train_tensors(words_input,
                                    tags, word_to_index,
                                    dataset.tag_to_index,
                                    suffix2id=dataset.suffix_to_index)
        # reset gradients, else they accumulate
        optimizer.zero_grad()

        acc_output = Variable(v_type(args.batch_size, len(input_tensor), dataset.get_num_tags()))
        # loop sequence
        for seq_n in range(len(input_tensor)):

            output = model(Variable(input_tensor[seq_n]),
                           Variable(cap_tensor[seq_n]),
                           Variable(suf_tensor[seq_n]))
            acc_output[:, seq_n, :] = output

        score = model.sentence_level_log_likelihood(acc_output, target_tensor)

        score.backward()
        optimizer.step()

        print('', end='\r')
        print("{:15} | Epoch {:5} Iteration {:5}\
              Score {:7.5f}".format(time_since(start),
                                    epoch,
                                    i,
                                    score.data[0]), end='')

    print("\n")

    # run test on dev
    model.set_batch_size(1)

    tagged_file = open("tmp/dev_colo.annotated", "w")
    for i, (words_input, tags) in enumerate(dev_loader):
        input_tensor, target_tensor, cap_tensor, suf_tensor = model.get_train_tensors(words_input, tags,
        word_to_index,dataset.tag_to_index,suffix2id=dataset.suffix_to_index)
        acc_output = Variable(v_type(1, len(input_tensor), dataset.get_num_tags()))
        # loop sequence
        for seq_n in range(len(input_tensor)):

            output = model(Variable(input_tensor[seq_n]),
                           Variable(cap_tensor[seq_n]),
                           Variable(suf_tensor[seq_n]))
            acc_output[:, seq_n, :] = output
        for batch_i, batch_out in enumerate(acc_output):
            best_score, best_path = model.viterbi_decode(batch_out)
            for i,tag_i in enumerate(best_path):
                word = words_input[i][args.window_size][batch_i]
                real_tag = dataset.index_to_tag(target_tensor[batch_i][i])
                predicted_tag = dataset.index_to_tag(tag_i)
                tagged_file.write("{} {} {}\n".format(word,real_tag,predicted_tag))

    tagged_file.close()
    acc, prec, rec, f1 = evaluate("tmp/dev_colo.annotated")
    if f1 > dev_best_f1:
        dev_best_f1 = f1
        dev_best_epoch = epoch
        # run test
        tagged_file = open("tmp/test_colo.annotated", "w")
        for i, (words_input, tags) in enumerate(test_loader):
            input_tensor, target_tensor, cap_tensor, suf_tensor = model.get_train_tensors(words_input, tags,
            word_to_index,dataset.tag_to_index,suffix2id=dataset.suffix_to_index)
            acc_output = Variable(v_type(1, len(input_tensor), dataset.get_num_tags()))
            # loop sequence
            for seq_n in range(len(input_tensor)):

                output = model(Variable(input_tensor[seq_n]),
                               Variable(cap_tensor[seq_n]),
                               Variable(suf_tensor[seq_n]))
                acc_output[:, seq_n, :] = output
            for batch_i, batch_out in enumerate(acc_output):
                best_score, best_path = model.viterbi_decode(batch_out)
                for i,tag_i in enumerate(best_path):
                    word = words_input[i][args.window_size][batch_i]
                    real_tag = dataset.index_to_tag(target_tensor[batch_i][i])
                    predicted_tag = dataset.index_to_tag(tag_i)
                    tagged_file.write("{} {} {}\n".format(word,real_tag,predicted_tag))

        tagged_file.close()
        acc, prec, rec, f1 = evaluate("tmp/test_colo.annotated")
        if f1 > test_best_f1:
            test_best_f1 = f1
            test_best_epoch = epoch


    print("Best dev F1 score {} at epoch {}".format(dev_best_f1, dev_best_epoch))
    print("Best test F1 score {} at epoch {}".format(test_best_f1, test_best_epoch))
