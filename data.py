from ntap.data import *
import collections
import random

class DemoData(Dataset):
    def __init__(self, source, demo_cols = None, demo_path=None, glove_path=None, mallet_path=None, tokenizer='wordpunct', vocab_size=5000,
            embed='glove', min_token=5, stopwords=None, stem=False,
            lower=True, max_len=512, include_nums=False,
            include_symbols=False, num_topics=100, lda_max_iter=500):

        Dataset.__init__(self, source, glove_path, mallet_path, tokenizer, vocab_size,
            embed, min_token, stopwords, stem,
            lower, max_len, include_nums,
            include_symbols, num_topics, lda_max_iter)

        if "username" in self.data.columns:
            self.annotators = sorted(set(self.data["username"]))
        else:
            columns = self.data.columns.tolist()
            columns.remove("text")
            self.annotators = sorted([int(x) for x in columns])

        if demo_path:
            self.__read_demo(demo_path, demo_cols)
        else:
            self.demo = None
        #self.data = self.data[self.data["username"].isin(self.annotators)]


    def __read_demo(self, demo_path, cols):
        demo_df = pd.read_csv(demo_path)
        #cols = list(demo_df.columns)
        #cols.remove("Username")
        self.demo = {anno: 0 for anno in self.annotators}
        self.demo_dim = len(cols)

        for i, row in demo_df.iterrows():
            self.demo[row["Username"]] = np.array([row[col] for col in cols])

        missing = np.random.uniform(0, 1, self.demo_dim)

        for i in range(max(self.annotators)):
            if i not in self.demo.keys():
                self.demo[i] = missing
        self.demo = np.array([self.demo[i] for i in sorted(self.demo.keys())])


    def batches(self, var_dict, batch_size, test, keep_ratio=None, idx=None):
        feed_dict = dict()

        if idx is None:
            idx = [i for i in range(self.num_sequences)]

        for (s, e) in self._Dataset__batch_indices(len(idx), batch_size):
            for var_name in var_dict:
                if var_name == 'word_inputs':
                    feed_dict[var_dict[var_name]] = self._Dataset__add_padding(
                        self.sequence_data[idx[s:e]])
                if var_name == 'sequence_length':
                    feed_dict[var_dict[var_name]] = self.sequence_lengths[idx[s:e]]
                if var_name == "annotators":
                    feed_dict[var_dict["annotators"]] = self._Dataset__annotators(idx[s:e])
                if var_name == "annotator":
                    feed_dict[var_dict[var_name]] = self.data["username"][idx[s:e]]
                if var_name == "gather":
                    feed_dict[var_dict["gather"]] = np.array([[self.annotators.index(anno), i]
                                                              for i, anno in
                    enumerate(self.data["username"][idx[s:e]])])
                if var_name == "DemoEmbeddingPlaceholder":
                    feed_dict[var_dict[var_name]] = self.demo
                if var_name.startswith('target'):
                    name = var_name.replace("target-", "")
                    if name not in self.targets:
                        raise ValueError("Target not in data: {}".format(name))
                    feed_dict[var_dict[var_name]] = self.targets[name][idx[s:e]]
                if var_name.startswith("mask"):
                    name = var_name.replace("mask-", "")
                    mask = self.targets[name][idx[s:e]]
                    mask[mask == 0] = True
                    mask[mask == 1] = True
                    mask[mask == -1] = False
                    feed_dict[var_dict[var_name]] = mask
                if test:
                    feed_dict[var_dict['keep_ratio']] = 0
                    continue  # no labels or loss weights
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


class MultiData(Dataset):
    def __init__(self, source, demo_path=None, glove_path=None, mallet_path=None,
                 tokenizer='wordpunct', vocab_size=5000,
            embed='glove', min_token=5, stopwords=None, stem=False,
            lower=True, max_len=512, include_nums=False,
            include_symbols=False, num_topics=100, lda_max_iter=500):

        Dataset.__init__(self, source, glove_path, mallet_path, tokenizer, vocab_size,
            embed, min_token, stopwords, stem,
            lower, max_len, include_nums,
            include_symbols, num_topics, lda_max_iter)

        columns = list(self.data.columns)
        columns.remove("text")
        self.annotators = sorted(columns)
        if demo_path:
            self.__read_demo(demo_path)

    def __read_demo(self, demo_path):
        demo_df = pd.read_csv(demo_path)
        cols = list(demo_df.columns)
        cols.remove("Username")
        self.demo = dict()
        self.demo_dim = len(cols)

        for i, row in demo_df.iterrows():
            self.demo[row["Username"]] = np.array([row[col] for col in cols])

        missing = np.random.randint(-3, 3, self.demo_dim)

        for i in range(max(self.annotators)):
            if i not in self.demo.keys():
                self.demo[i] = missing
        self.demo = np.array([self.demo[i] for i in sorted(self.demo.keys())])

    def __annotators(self, batch):
        _b = batch[0]
        _target_ids = {an: i for i, an in enumerate(self.target_names.keys())}
        _anno = [_target_ids[_target] for _target in self.target_names.keys()
                 if re.match("[0-9]+", _target) and self.targets[_target][_b] != 2]
        return np.array(_anno)

    def batches(self, var_dict, batch_size, test, keep_ratio=None, idx=None):
        feed_dict = dict()

        if idx is None:
            idx = [i for i in range(self.num_sequences)]

        for (s, e) in self._Dataset__batch_indices(len(idx), batch_size):
            for var_name in var_dict:
                if var_name == 'word_inputs':
                    feed_dict[var_dict[var_name]] = self._Dataset__add_padding(
                        self.sequence_data[idx[s:e]])
                if var_name == 'sequence_length':
                    feed_dict[var_dict[var_name]] = self.sequence_lengths[idx[s:e]]
                if var_name == "annotators":
                    feed_dict[var_dict["annotators"]] = self._Dataset__annotators(idx[s:e])
                if var_name == "annotator":
                    feed_dict[var_dict[var_name]] = self.data["username"][idx[s:e]]
                if var_name == "gather":
                    feed_dict[var_dict["gather"]] = np.array([[self.annotators.index(anno), i]
                                                              for i, anno in
                                                              enumerate(self.data["username"][idx[s:e]])])
                if var_name == "DemoEmbeddingPlaceholder":
                    feed_dict[var_dict[var_name]] = self.demo
                if var_name.startswith('target'):
                    name = var_name.replace("target-", "")
                    if name not in self.targets:
                        raise ValueError("Target not in data: {}".format(name))
                    feed_dict[var_dict[var_name]] = self.targets[name][idx[s:e]]
                if test:
                    feed_dict[var_dict['keep_ratio']] = 1.0
                    continue  # no labels or loss weights
                if var_name.startswith("weights"):
                    name = var_name.replace("weights-", "")
                    if name not in self.weights:
                        raise ValueError("Weights not found in data")
                    feed_dict[var_dict[var_name]] = np.array(self.weights[name])
                    print(np.array(self.weights[name]))
                    exit(1)
                if var_name == 'keep_ratio':
                    if keep_ratio is None:
                        raise ValueError("Keep Ratio for RNN Dropout not set")
                    feed_dict[var_dict[var_name]] = keep_ratio
            yield feed_dict

