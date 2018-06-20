import torch
from rnn import RNN
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import time
import math
import random
from type_factory import LongTensor,FloatTensor
import pickle
import argparse

parser = argparse.ArgumentParser(description='Toy Char-RNN')
parser.add_argument("-s","--sanity", help="Run sanity check",action='store_true')
args = parser.parse_args()

filename="i_wonderland.txt"

if args.sanity:
    raw_text = "hello world! life is pain"
    seq_length = 2
else:
    raw_text = open(filename).read()
    seq_length = 25

#create char mapping
chars = sorted(list(set(raw_text)))
char_to_int = dict((c,i) for i,c in enumerate(chars))
int_to_char = dict((i,c) for i,c in enumerate(chars))

#data summary
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: {}, Vocabulary size: {}.".format(n_chars,n_vocab))

#prepare data input
dataX = [] #network input data
dataY = [] #network target variable
for i in range(0,n_chars-seq_length,1):
    seq = raw_text[i:i+seq_length]
    target = raw_text[i+seq_length]
    dataX.append(seq)
    dataY.append(target)
n_patterns = len(dataX)
print("Total training examples: {}.".format(n_patterns))

#transform inputs to tensors
def letter_to_tensor(c):
    return LongTensor()([char_to_int[c]])
def sequence_to_tensor(s):
    tensor = FloatTensor()(len(s),1,n_vocab).fill_(0)
    for i,letter in enumerate(s):
        tensor[i][0][char_to_int[letter]] = 1
    return tensor

n_hidden = 512
rnn = RNN(n_vocab, n_hidden, n_vocab) # input is chars, output is probability of next chars
print("Neural net instanciated.")
if torch.cuda.is_available():
    rnn.cuda()

def char_from_output(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    char_i = top_i[0][0]
    return int_to_char[char_i], char_i

def randomTrainingExample():
    i = random.randint(0,len(dataX)-1)
    return dataX[i],dataY[i],Variable(sequence_to_tensor(dataX[i])),Variable(letter_to_tensor(dataY[i]))

learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn
criterion = nn.NLLLoss()
optimizer = optim.SGD(rnn.parameters(), lr=learning_rate)


def train(target_tensor, input_tensor):
    hidden = rnn.initHidden()

    #reset gradients, else they accumulate
    optimizer.zero_grad()
    rnn.zero_grad()

    #feed all chars in the sequence
    for i in range(input_tensor.size()[0]):
        output, hidden = rnn(input_tensor[i], hidden)

    loss = criterion(output, target_tensor)
    loss.backward()

    # Clipping to avoid exploding gradient
    #torch.nn.utils.clip_grad_norm(rnn.parameters(), 0.25)
    # Update parameters(weights)
    optimizer.step()

    return output, loss.data[0]

if args.sanity:
    n_iters = 10000
    print_every = 100
    plot_every = 100
else:
    n_iters = 1000000
    print_every = 5000
    plot_every = 10000

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()
for iter in range(1, n_iters + 1):
    sequence,target,sequence_tensor,target_tensor = randomTrainingExample()
    output, loss = train(target_tensor, sequence_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = char_from_output(output)
        correct = '✓' if guess == target else '✗ (%s)' % target
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, sequence, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

if args.sanity:
    print("-"*20)
    print("Testing generation")
    hidden = rnn.initHidden()
    start = Variable(sequence_to_tensor("he"))
    _, hidden = rnn(start[0], hidden)
    output, hidden = rnn(start[1], hidden)
    input = "he"
    print("{}".format(input),end='')
    for i in range(50):
        hidden = rnn.initHidden()
        start = Variable(sequence_to_tensor(input))
        _, hidden = rnn(start[0], hidden)
        output, hidden = rnn(start[1], hidden)
        guess, _ = char_from_output(output)
        input = input[1]+guess
        print("{}".format(guess),end='')

torch.save(rnn.state_dict(),".saved_model")
