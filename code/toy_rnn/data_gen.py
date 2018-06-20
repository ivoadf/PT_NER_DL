from torch.utils.data import Dataset
from type_factory import LongTensor,FloatTensor

class CharDataset(Dataset):
    def __init__(self,filename,seq_length,sanity=False):
        if sanity:
            raw_text = "hello world!"
            seq_length = 2
        else:
            with open(filename,"r") as f:
                raw_text = f.read(100)
        #create char mapping
        chars = sorted(list(set(raw_text)))
        self.char_to_int = dict((c,i) for i,c in enumerate(chars))
        self.int_to_char = dict((i,c) for i,c in enumerate(chars))

        #data summary
        n_chars = len(raw_text)
        self.vocab_size = len(chars)

        #prepare data input
        self.dataX = [] #network input data
        self.dataY = [] #network target variable
        for i in range(0,n_chars-seq_length,1):
            seq = raw_text[i:i+seq_length]
            target = raw_text[i+seq_length]
            self.dataX.append(seq)
            self.dataY.append(target)
        self.len = len(self.dataX)
    def __getitem__(self,i):
        return self.sequence_to_tensor(self.dataX[i]),self.letter_to_tensor(self.dataY[i])
    def __len__(self):
        return self.len

    #transform inputs to tensors
    def letter_to_tensor(self,c):
        return LongTensor()([self.char_to_int[c]])
    def sequence_to_tensor(self,s):
        tensor = FloatTensor()(len(s),self.vocab_size).fill_(0)
        for i,letter in enumerate(s):
            tensor[i][self.char_to_int[letter]] = 1
        return tensor
