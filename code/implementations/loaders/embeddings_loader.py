import numpy as np
UNK_EMBEDDING_INDEX = 0

class EmbeddingsLoader():
    """
    Loads embeddings that follow the format:
    WORD1 E11 E12 .... E1N
    WORD2 E21 E22 .... E2N
    """
    def __init__(self,filename):
        f = open(filename,'r')
        self.word_to_id = dict()
        self.embeddings = []
        self.emb_dim = -1

        for i,line in enumerate(f):
            line_s = line.split()
            if len(line_s) < 3:
                continue
            word = line_s[0]

            while line_s[1].isalpha():
                word += " "+line_s[1]
                del line_s[1]

            self.word_to_id[word] = i
            self.embeddings.append([float(n) for n in line_s[1:]])
        f.close()

    def word_to_index(self,word):
        if word.lower() in self.word_to_id:
            return self.word_to_id[word.lower()]
        elif 'UNK' in self.word_to_id or '</s>' in self.word_to_id:
            return UNK_EMBEDDING_INDEX
        else:
            return np.zeros(len(self.embeddings[0]))

    def get_weights(self):
        return self.embeddings

    def get_dimensions(self):
        return len(self.embeddings),len(self.embeddings[0])
