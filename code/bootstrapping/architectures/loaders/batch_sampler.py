from torch.utils.data.sampler import Sampler


class BatchSampler(Sampler):

    def __init__(self, data_source, min_batch_size=1, enforce=False):
        self.data_source = data_source
        self.sentence_lengths = data_source.get_sentence_length()
        max_batch_size = 1000
        for i in self.sentence_lengths:
            if len(self.sentence_lengths[i]) < max_batch_size:
                max_batch_size = len(self.sentence_lengths[i])
        self.batch_index_list = []
        if max_batch_size < min_batch_size or enforce:
            max_batch_size = min_batch_size
        for i in self.sentence_lengths:
            tmp = 0
            while tmp < len(self.sentence_lengths[i]):
                if tmp+max_batch_size > len(self.sentence_lengths[i]):
                    break
                self.batch_index_list.append(self.sentence_lengths[i][tmp:tmp+max_batch_size])
                tmp += max_batch_size

    def __iter__(self):
        return iter(self.batch_index_list)

    def __len__(self):
        return len(self.batch_index_list)
