import sys
sys.path.append('.')

from ntap.data import Dataset
from model import *

def initialize_dataset(data_dir):
    data = Dataset(data_dir)
    data.set_params(vocab_size=10000,
                    mallet_path = "/home/aida/Data/mallet/mallet-2.0.8/bin/mallet",
                    glove_path = "/Users/aidadavani/Desktop/glove.6B.300d.txt")
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
    result = model.CV(data, num_epochs=10, num_folds=10)
    result.summary()
    #model.train(data, num_epochs=10, model_path="save/model")


if __name__== '__main__':
    data = initialize_dataset("posts.csv")
    model = initialize_model(data)
    train_model(model, data)