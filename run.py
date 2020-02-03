import sys
sys.path.append('.')

from model import *
from data import *
import argparse

def initialize_dataset(mode):
    if mode == "annotator":
        #data = DemoData("Data/annotations_id.csv", demo_path="Data/demo_clean.csv")
        data = MultiData("Data/sub_posts.csv")
    else:
        data = DemoData("Data/annotations_id.csv", demo_path="Data/demo_clean.csv")
        #data = MultiData("Data/sub_posts.csv", demo_path="Data/demo_clean.csv")


    data.set_params(vocab_size=10000,
                    mallet_path = "/home/aida/Data/mallet/mallet-2.0.8/bin/mallet",
                    glove_path = "/Users/aidadavani/Desktop/glove.6B.300d.txt")
    data.clean("text")
    return data

def initialize_model(data, mode):
    if mode == "annotator":
        #model = Annotator("hate ~ seq(text)",
        #            rnn_dropout=0.3, cell="biGRU",
        #            embedding_source="glove", data=data, optimizer='adam',
        #            learning_rate=0.07, hidden_size=128)
        cols = list(data.data.columns)
        cols.remove("text")

        iv = "+".join([str(x) for x in cols])
        model = xAnnotator(iv + " ~ seq(text)",
                    rnn_dropout=0.4, cell="biGRU",
                    embedding_source="glove", data=data, optimizer='adam',
                    learning_rate=0.0003, hidden_size=128)
        # 10 epochs
    else:
        model = AnnotatorDemo("hate ~ seq(text)",
                          rnn_dropout=0.5, hidden_size=50, cell="biGRU",
                          embedding_source="glove", data=data, optimizer='adam',
                          learning_rate=0.0005)
        """
        cols = list(data.data.columns)
        cols.remove("text")

        iv = "+".join([str(x) for x in cols])
        model = xAnnotatorDemo(iv + " ~ seq(text)",
                    rnn_dropout=0.4, cell="biGRU",
                    embedding_source="glove", data=data, optimizer='adam',
                    learning_rate=0.0003, hidden_size=128)
        """
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
#accuracy f1 kappa precision recall
# Mean  0.958087  0.804051  0.781523   0.680086  0.985083

