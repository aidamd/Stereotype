import sys
sys.path.append('.')

from model import *
from data import *

def initialize_dataset(data_dir):
    data = MultiData(data_dir)
    data.set_params(vocab_size=10000,
                    mallet_path = "/home/aida/Data/mallet/mallet-2.0.8/bin/mallet",
                    glove_path = "/home/aida/Data/word_embeddings/GloVe/glove.840B.300d.txt")
    data.clean("text")
    return data

def initialize_model(data):
    annotators = data.data.columns.tolist()
    annotators.remove("id")
    annotators.remove("text")
    dv = "+".join(a for a in annotators)

    model = Multi(dv + " ~ seq(text)",
            rnn_dropout=0.2, hidden_size=100, cell="biGRU",
            embedding_source="glove", data=data, optimizer='adam',
            learning_rate=0.0001)
    return model

def train_model(model, data):
    result = model.CV(data, num_epochs=20, num_folds=10)
    result.summary()
    #model.train(data, num_epochs=10, model_path="save/model")


if __name__== '__main__':
    data = initialize_dataset("posts.csv")
    model = initialize_model(data)
    train_model(model, data)