import sys
sys.path.append('.')

from model import *
from data import *
import argparse

def initialize_dataset(mode):
    data = DemoData("Data/sub_posts.csv",
                    demo_path="Data/demo_clean.csv" if mode == "demo" else None)

    data.set_params(vocab_size=10000,
                    mallet_path = "/home/aida/Data/mallet/mallet-2.0.8/bin/mallet",
                    glove_path = "/home/aida/Data/word_embeddings/GloVe/glove.840B.300d.txt")
    data.clean("text")
    return data

def initialize_model(data, mode):
    dv = "+".join([str(col) for col in data.annotators])
    model = MultiModel(dv + " ~ seq(text)",
                          rnn_dropout=0.5, hidden_size=64, cell="biGRU",
                          embedding_source="glove", data=data, optimizer='adam',
                          learning_rate=0.0005)
    return model


def train_model(model, data):
    result = model.CV(data, num_epochs=10, num_folds=10, batch_size=512)
    result.summary()

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="mode represents each of the three studies, "
                                       "it can be agree, annotator, or annotator_demo")

    args = parser.parse_args()
    data = initialize_dataset(args.mode)
    model = initialize_model(data, args.mode)
    train_model(model, data)

# study 1
# accuracy f1 kappa precision recall
# Mean  0.958087  0.804051  0.781523   0.680086  0.985083

# annotation:
# accuracy f1 kappa precision recall
# Mean  0.871081  0.558240  0.484313   0.641480  0.497457


# demo:
# accuracy f1 kappa precision recall
# Mean  0.877147  0.566522  0.497966   0.689727  0.484083


