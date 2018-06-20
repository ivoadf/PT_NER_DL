from torch.utils.data import Dataset


class ConllStyleDataset(Dataset):
    """
    Processes file in CONLL style
    If window size defined:
        dataX = (num_sentences,sentence_length,window_size)
        dataY = (num_sentences,sentence_length)
    If sequence if true:
        dataX = (num_sentences,sentence_length)
        dataY = (num_sentences,sentence_length)
    """
    def __init__(self, dataset, suffix_size, vocab_size,
                 compute_dicts, window_size=None, sequence=None, start=0, end=1):
        self.dataset = dataset
        split_s = int(len(self.dataset)*start)
        split_e = int(len(self.dataset)*end)
        self.dataset = self.dataset[split_s:split_e]
        self.suffix_size = suffix_size
        self.vocab_size = vocab_size

        self.dataX = []
        self.dataY = []
        self.tags_set = set()
        self.suffix_set = set()
        word_dict = dict()
        self.sentence_length = {}
        if window_size is not None:
            sentence_windows = []
            sentence_tags = []
            tmpX = ['pad']*window_size
            tmpY = ['O']*window_size
            counter = window_size
            for line in self.dataset:
                line_s = line.split()

                # end of document/sentence
                if len(line_s) == 0:
                    if len(sentence_windows) > 100 or len(sentence_windows) < 3:  # skip small and large sentences
                        sentence_windows = []
                        sentence_tags = []
                        tmpX = ['pad']*window_size
                        tmpY = ['O']*window_size
                        counter = window_size
                        continue
                    if len(tmpX) == window_size+1:  # single token sentence
                        tmpX = ['pad']*window_size
                        tmpY = ['O']*window_size
                        counter = window_size
                        continue

                    tmpX += ['pad']*window_size
                    tmpY += ['O']*window_size

                    # add tail windows with padding before reseting
                    for i in range(window_size):
                        sentence_windows.append(list(tmpX[i:i+1+2*window_size]))
                        sentence_tags.append(tmpY[i+window_size])

                    if len(sentence_windows) not in self.sentence_length:
                        self.sentence_length[len(sentence_windows)] = [len(self.dataX)]
                    else:
                        self.sentence_length[len(sentence_windows)].append(len(self.dataX))
                    self.dataX.append(list(sentence_windows))
                    self.dataY.append(list(sentence_tags))
                    sentence_windows = []
                    sentence_tags = []
                    tmpX = ['pad']*window_size
                    tmpY = ['O']*window_size
                    counter = window_size
                    continue

                word, tag = line_s[0], line_s[-1]
                self.tags_set.add(tag)
                self.suffix_set.add(word[:-suffix_size])
                if compute_dicts:
                    if word.lower() in word_dict:
                        word_dict[word.lower()] += 1
                    else:
                        word_dict[word.lower()] = 1

                tmpX.append(word)
                tmpY.append(tag)
                counter += 1
                if counter == 1+window_size*2:
                    counter -= 1
                    sentence_windows.append(list(tmpX))
                    sentence_tags.append(tmpY[window_size])  # only tag is from center word
                    del tmpX[0]
                    del tmpY[0]
        elif sequence:
            tmpX = []
            tmpY = []
            for line in self.dataset:
                line_s = line.split()

                # end of document/sentence
                if len(line_s) == 0:
                    if len(tmpX) > 100 or len(tmpX) < 3:  # or len(set(tmpY)) < 2:  # skip small and large sentences and sentences that do not have an entity
                        tmpX = []
                        tmpY = []
                        continue
                    if len(tmpX) not in self.sentence_length:
                        self.sentence_length[len(tmpX)] = [len(self.dataX)]
                    else:
                        self.sentence_length[len(tmpX)].append(len(self.dataX))
                    self.dataX.append(list(tmpX))
                    self.dataY.append(list(tmpY))
                    tmpX = []
                    tmpY = []
                    continue

                word, tag = line_s[0], line_s[-1]
                self.tags_set.add(tag)
                self.suffix_set.add(word[:-suffix_size])
                if compute_dicts:
                    if word.lower() in word_dict:
                        word_dict[word.lower()] += 1
                    else:
                        word_dict[word.lower()] = 1

                tmpX.append(word)
                tmpY.append(tag)
        self.tag_to_id = {}
        self.id_to_tag = {}
        self.suffix_to_id = {}
        self.word_to_id = {}
        for i, tag in enumerate(sorted(self.tags_set)):
            self.tag_to_id[tag] = i
            self.id_to_tag[i] = tag
        for i, suffix in enumerate(sorted(self.suffix_set)):
            self.suffix_to_id[suffix] = i

        if compute_dicts:
            assert vocab_size <= len(word_dict)
            word_list = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
            word_list = word_list[:vocab_size-1]
            for i, (key, value) in enumerate(word_list):
                self.word_to_id[key] = i

    def __getitem__(self, i):
        return self.dataX[i], self.dataY[i]

    def __len__(self):
        return len(self.dataX)

    def get_num_tags(self):
        return len(self.tags_set)

    def tag_to_index(self, tag):
        return self.tag_to_id[tag]

    def index_to_tag(self, id):
        return self.id_to_tag[id]

    def suffix_to_index(self, w):
        s = w[:-self.suffix_size]
        if s in self.suffix_set:
            return self.suffix_to_id[s]
        else:
            return len(self.suffix_set)  # last id is for unknown

    def get_num_suffix(self):
        return len(self.suffix_set)+1

    def get_num_words(self):
        return len(self.word_to_id)+1

    def word_to_index(self, word):
        if word.lower() in self.word_to_id:
            return self.word_to_id[word.lower()]
        else:
            return self.vocab_size-1  # last id for unknown

    def get_tag_dict(self):
        r = dict()
        for tag in self.dataY:
            if type(tag) is list:
                for t in tag:
                    if t not in r:
                        r[t] = 1
                    else:
                        r[t] += 1
            else:
                if tag not in r:
                    r[tag] = 1
                else:
                    r[tag] += 1
        return r
    """
    Dictionary containing indexes of sentences with same length
    """
    def get_sentence_length(self):
        return self.sentence_length


class ConllWindowFromFile(ConllStyleDataset):
    def __init__(self, filename, window_size, suffix_size=3,
                 vocab_size=100000, compute_dicts=False):
        dataset = open(filename, "r").readlines()
        super(ConllWindowFromFile, self).__init__(dataset,
                                                  window_size=window_size,
                                                  suffix_size=suffix_size,
                                                  vocab_size=vocab_size,
                                                  compute_dicts=compute_dicts)


class ConllWindowFromRaw(ConllStyleDataset):
    def __init__(self, raw, window_size, suffix_size=3, vocab_size=10):
        super(ConllWindowFromRaw, self).__init__(raw,
                                                 window_size=window_size,
                                                 suffix_size=suffix_size,
                                                 vocab_size=vocab_size,
                                                 compute_dicts=True)


def sanity_window_dataset(window_size, vocab_size):
    # raw = ["O O", "José B-Pessoa", "Barbosa E-Pessoa", "comprou O",
    # "500 S-Valor", "acções O", "da O", "Apple S-Organizacao",
    # "em O", "Fevereiro B-Data", "de I-Data", "1995 E-Data", ". O", "\n"]
    lines = open("../preprocessing/i_GC_MiniHAREM_CONLL.txt", "r").readlines()
    raw = lines[:1500]
    raw2 = lines[1500:2000]
    return ConllWindowFromRaw(raw, window_size, vocab_size=vocab_size),\
        ConllWindowFromRaw(raw2, window_size, vocab_size=vocab_size)


class ConllSequenceFromFile(ConllStyleDataset):
    def __init__(self, filename, suffix_size=3, vocab_size=100000, compute_dicts=False):
        dataset = open(filename, "r").readlines()
        super(ConllSequenceFromFile, self).__init__(dataset,
                                                    sequence=True,
                                                    suffix_size=suffix_size,
                                                    vocab_size=vocab_size,
                                                    compute_dicts=compute_dicts)


class ConllSequenceFromRaw(ConllStyleDataset):

    def __init__(self, raw, suffix_size=3, vocab_size=10, compute_dicts=True):
        super(ConllSequenceFromRaw, self).__init__(raw,
                                                   sequence=True,
                                                   suffix_size=suffix_size,
                                                   vocab_size=vocab_size,
                                                   compute_dicts=compute_dicts)


def sanity_sequence_dataset(vocab_size):
    lines = open("../preprocessing/i_GC_MiniHAREM_CONLL.txt", "r").readlines()
    # raw = ["O O", "João S-Pessoa", "nasceu O", "em O", "Pombal S-Local", '. O', '\n',
    #        "O O", "Paulo S-Pessoa", "nasceu O", "em O", "Espinho S-Local", '. O', '\n',
    #        "O O", "Manuel S-Pessoa", "nasceu O", "em O", "Lisboa S-Local", '. O', '\n',
    #        "O O", "Pedro S-Pessoa", "nasceu O", "em O", "Sintra S-Local", '. O', '\n',
    #        "O O", "Ricardo S-Pessoa", "nasceu O", "em O", "Faro S-Local", '. O', '\n']
    # raw2 = ["O O", "José S-Pessoa", "nasceu O", "em O", "Leiria S-Local", '. O', '\n']
    raw = lines[:50]
    raw2 = lines[:50]
    return ConllSequenceFromRaw(raw, vocab_size=vocab_size), \
        ConllSequenceFromRaw(raw2, vocab_size=vocab_size, compute_dicts=False)
