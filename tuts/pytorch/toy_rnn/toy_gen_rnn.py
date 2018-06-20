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

USE_GPU = True

filename="i_wonderland.txt"
raw_text = open(filename).read()

#create char mapping
chars = sorted(list(set(raw_text)))
char_to_int = dict((c,i) for i,c in enumerate(chars))
int_to_char = dict((i,c) for i,c in enumerate(chars))

#data summary
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: {}, Vocabulary size: {}.".format(n_chars,n_vocab))

#prepare data input
seq_length = 100
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
    return LongTensor(USE_GPU)([char_to_int[c]])
def sequence_to_tensor(s):
    tensor = FloatTensor(USE_GPU)(len(s),1,n_vocab).fill_(0)
    for i,letter in enumerate(s):
        tensor[i][0][char_to_int[letter]] = 1
    return tensor

n_hidden = 128
rnn = RNN(n_vocab, n_hidden, n_vocab) # input is chars, output is probability of next char
if USE_GPU:
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
    hidden = rnn.initHidden(USE_GPU)

    #reset gradients, else they accumulate
    optimizer.zero_grad()

    #feed all chars in the sequence
    for i in range(input_tensor.size()[0]):
        output, hidden = rnn(input_tensor[i], hidden)

    loss = criterion(output, target_tensor)
    loss.backward()

    # Update parameters(weights)
    optimizer.step()

    return output, loss.data[0]


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

with open('losses.pickle','wb') as f:
    pickle.dump(all_losses,f)

torch.save(rnn,'fullmodel')
torch.save(rnn.state_dict(),'modelparams')

#the_model = TheModelClass(*args, **kwargs)
#the_model.load_state_dict(torch.load(PATH))
#or
#the_model = torch.load(PATH)
