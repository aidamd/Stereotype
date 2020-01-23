import sys
sys.path.append('.')

from model import *
from data import *
import argparse

def initialize_dataset(mode):
    if mode == "annotator":
        data = MultiData("Data/posts.csv")
    elif mode == "agree":
        data = Dataset("Data/posts.csv")
    else:
        data = DemoData("Data/annotations_id.csv", demo_path="Data/IAT_clean.csv")

    data.set_params(vocab_size=10000,
                    mallet_path = "/home/aida/Data/mallet/mallet-2.0.8/bin/mallet",
                    glove_path = "/home/aida/Data/word_embeddings/GloVe/glove.840B.300d.txt")
    data.clean("text")
    return data

def initialize_model(data, mode):
    if mode == "annotator":
        annotators = data.data.columns.tolist()
        annotators.remove("id")
        annotators.remove("text")
        dv = "+".join(a for a in annotators)
        model = Annotator(dv + " ~ seq(text)",
                    rnn_dropout=0.2, hidden_size=100, cell="biGRU",
                    embedding_source="glove", data=data, optimizer='adam',
                    learning_rate=0.0001)
    elif mode == "agree":
        model = RNN("agreement ~ seq(text)",
                    rnn_dropout=0.2, hidden_size=100, cell="biGRU",
                    embedding_source="glove", data=data, optimizer='adam',
                    learning_rate=0.0001)
    else:
        model = AnnotatorDemo("hate ~ seq(text)",
                          rnn_dropout=0.2, hidden_size=100, cell="biGRU",
                          embedding_source="glove", data=data, optimizer='adam',
                          learning_rate=0.0001)
    return model

def train_model(model, data):
    result = model.CV(data, num_epochs=15, num_folds=10)
    result.summary()

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode")

    args = parser.parse_args()
    data = initialize_dataset(args.mode)
    model = initialize_model(data, args.mode)
    train_model(model, data)
