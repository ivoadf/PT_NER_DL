import torch.nn as nn
import torch
from torch.autograd import Variable

"""
Carnegie Mellon Char CNN, paper = Ma2016
"""
CHAR_EMB_DIM = 30
CNN_WINDOW_SIZE = 3
CNN_FILTERS = 30
DROPOUT = 0.5


class CharCNN(nn.Module):

    def __init__(self, number_chars, gpu=False):
        super(CharCNN, self).__init__()
        self.number_chars = number_chars

        self.gpu = gpu

        if self.gpu and torch.cuda.is_available():
            self.long_type = torch.cuda.LongTensor
            self.float_type = torch.cuda.FloatTensor
        else:
            self.long_type = torch.LongTensor
            self.float_type = torch.FloatTensor

        self.dropout_layer = nn.Dropout(p=DROPOUT)
        self.char_embedings = nn.Embedding(number_chars, CHAR_EMB_DIM)

        """
            Input to Conv2D: (N,C,H,W)
            N - batch size
            C - number input channels
            H - height
            W - width

            Output: (N,C,H,W)
            N - batch size
            C - number output channels
            H,W - depend on parameters like padding, kernel size, stride and dillation

            kernel_size is [CNN_WINDOW_SIZE,CHAR_EMB_DIM]
            to obtain output of [1,CNN_FILTER,NUM_CHARS,1] to then max over
        """

        self.char_convolution = nn.Conv2d(in_channels=1, out_channels=CNN_FILTERS,
                                          kernel_size=(CNN_WINDOW_SIZE, CHAR_EMB_DIM))

    """
        chars: (N,W)
        N - batch_num
        W - indices of chars
    """
    def forward(self, chars):
        o = self.char_embedings(chars)
        o = o.view((1, 1, -1, CHAR_EMB_DIM))
        o = self.dropout_layer(o)
        o = self.char_convolution(o)
        o = o.view(1, 1, CNN_FILTERS, -1)
        o, _ = torch.max(o, dim=3)
        o = o.view(CNN_FILTERS)
        return o

    def word_to_char_var(self, word, char_to_id):
        num_padding_chars = (CNN_WINDOW_SIZE-1)*2
        char_tensor = self.long_type(1, len(word)+num_padding_chars)
        pointer = 0
        for _ in range(int(num_padding_chars/2)):
            char_tensor[0][pointer] = char_to_id(_, padding=True)
            pointer += 1
        for c in word:
            char_tensor[0][pointer] = char_to_id(c)
            pointer += 1
        for _ in range(int(num_padding_chars/2)):
            char_tensor[0][pointer] = char_to_id(_, padding=True)
            pointer += 1
        return Variable(char_tensor)
