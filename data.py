from ntap.data import *

class MultiData(Dataset):

    def __encode(self, column):
        self.__truncate_count = 0
        self.__pad_count = 0
        self.__unk_count = 0
        self.__token_count = 0
        tokenized = [None for _ in range(len(self.data))]

        self.sequence_lengths = list()

        for i, (_, string) in enumerate(self.data[column].iteritems()):
            tokens = self.__tokenize_doc(string)
            self.sequence_lengths.append(len(tokens))
            tokenized[i] = self.__encode_doc(tokens)

        # self.max_len = max(self.sequence_lengths)

        print("Encoded {} docs".format(len(tokenized)))
        print("{} tokens lost to truncation".format(self.__truncate_count))
        # print("{} padding tokens added".format(self.__pad_count))
        print("{:.3%} tokens covered by vocabulary of size {}".format(
            (self.__token_count - self.__unk_count) / self.__token_count, len(self.vocab)))
        self.sequence_data = np.array(tokenized)
        self.num_sequences = len(tokenized)
        self.sequence_lengths = np.array(self.sequence_lengths, dtype=np.int32)
        self.data = self.data.reset_index()

    def __annotators(self, batch):
        _b = batch[0]
        _anno = [int(_target) for _target in self.target_names.keys() if self.targets[_target][_b] != 2]
        return np.array(_anno)

    def __high_batch_indices(self, size, batch_size):
        mapping = self.data[list(self.target_names.keys())].replace(0, 1)
        for name, group in mapping.groupby(list(self.target_names.keys())):
            for i in range(0, group.shape[0], batch_size):
                yield group.iloc[i: min(i + batch_size, group.shape[0]):].index

    def high_batches(self, var_dict, batch_size, test, keep_ratio=None, idx=None):
        feed_dict = dict()

        if idx is None:
            idx = [i for i in range(self.num_sequences)]

        for sub_idx in self.__high_batch_indices(len(idx), batch_size):
            for var_name in var_dict:
                if var_name == 'word_inputs':
                    feed_dict[var_dict[var_name]] = self.__add_padding(self.sequence_data[sub_idx])
                if var_name == 'sequence_length':
                    feed_dict[var_dict[var_name]] = self.sequence_lengths[sub_idx]
                if var_name == "annotators":
                    feed_dict[var_dict["annotators"]] = self.__annotators(sub_idx)
                if test:
                    feed_dict[var_dict['keep_ratio']] = 1.0
                    continue  # no labels or loss weights
                if var_name.startswith('target'):
                    name = var_name.replace("target-", "")
                    if name not in self.targets:
                        raise ValueError("Target not in data: {}".format(name))
                    feed_dict[var_dict[var_name]] = self.targets[name][sub_idx]
                if var_name.startswith("weights"):
                    name = var_name.replace("weights-", "")
                    if name not in self.weights:
                        raise ValueError("Weights not found in data")
                    feed_dict[var_dict[var_name]] = np.array(self.weights[name])
                if var_name == 'keep_ratio':
                    if keep_ratio is None:
                        raise ValueError("Keep Ratio for RNN Dropout not set")
                    feed_dict[var_dict[var_name]] = keep_ratio
            yield feed_dict