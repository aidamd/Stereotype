from ntap.data import *

class MultiData(Dataset):

    def encode_targets(self, columns, var_type='categorical', normalize=None,
                       encoding='one-hot', reset=False):
        Dataset.encode_targets(self, columns, encoding='labels')
        length = len([x for x in self.targets[columns] if x != 2])
        self.weights[columns] = [(length - sum(self.targets[columns] == name)) /
                           length for name in [0, 1]]
        self.weights[columns].append(0)

    def __annotators(self, batch):
        _b = batch[0]
        _anno = [int(_target) for _target in self.target_names.keys() if self.targets[_target][_b] != 2]
        return np.array(_anno)

    def __high_batch_indices(self, idx, batch_size):
        mapping = self.data.iloc[idx][list(self.target_names.keys())].replace(0, 1)
        for name, group in mapping.groupby(list(self.target_names.keys())):
            for i in range(0, group.shape[0], batch_size):
                yield group.iloc[i: min(i + batch_size, group.shape[0]):].index

    def batches(self, var_dict, batch_size, test, keep_ratio=None, idx=None):
        feed_dict = dict()

        if idx is None:
            idx = [i for i in range(self.num_sequences)]

        for sub_idx in self.__high_batch_indices(idx, batch_size):
            for var_name in var_dict:
                if var_name == 'word_inputs':
                    feed_dict[var_dict[var_name]] = self._Dataset__add_padding(self.sequence_data[sub_idx])
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