import sys
sys.path.append('.')

from model import *
from data import *
import argparse

def initialize_dataset(mode):
    if mode == "annotator":
        #data = DemoData("Data/annotations_id.csv", demo_path="Data/demo_clean.csv")
        data = AnnoData("Data/sub_annotations.csv")
    else:
        data = DemoData("Data/sub_annotations.csv", demo_path="Data/demo_clean.csv")
        #data = MultiData("Data/sub_posts.csv", demo_path="Data/demo_clean.csv")


    data.set_params(vocab_size=10000,
                    mallet_path = "/home/aida/Data/mallet/mallet-2.0.8/bin/mallet",
                    glove_path = "/home/aida/Data/word_embeddings/GloVe/glove.840B.300d.txt")
    data.clean("text")
    return data

def initialize_model(data, mode):
    if mode == "annotator":
        model = AnnotatorDemo("hate ~ seq(text)",
                              rnn_dropout=0.5, hidden_size=64, cell="biGRU",
                              embedding_source="glove", data=data, optimizer='adam',
                              learning_rate=0.0005)
    else:
        model = AnnotatorDemo("hate ~ seq(text)",
                          rnn_dropout=0, hidden_size=64, cell="biGRU",
                          embedding_source="glove", data=data, optimizer='adam',
                          learning_rate=0.0001)

    return model

def train_model(model, data):
    result = model.CV(data, num_epochs=10, num_folds=5, batch_size=512)
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

