import sys
sys.path.append('.')

from model import *
from data import *
import argparse

def initialize_dataset(mode):
    if mode == "demo":
        data_path = "Data/sub_annotations.csv"
    else:
        data_path = "Data/sub_posts.csv"
    data = DemoData(data_path,
                    demo_path="Data/demo_clean.csv" if mode == "demo" else None)

    data.set_params(vocab_size=10000,
                    mallet_path = "/home/aida/Data/mallet/mallet-2.0.8/bin/mallet",
                    glove_path = "/home/aida/Data/word_embeddings/GloVe/glove.840B.300d.txt")
    data.clean("text")
    return data

def initialize_model(data, mode):
    if mode == "demo":
        model = AnnotatorDemo("hate ~ seq(text)",
                          rnn_dropout=0.5, hidden_size=50, cell="biGRU",
                          embedding_source="glove", data=data, optimizer='adam',
                          learning_rate=0.0005)
    else:
        dv = "+".join([str(col) for col in data.annotators])
        model = MultiModel(dv + " ~ seq(text)",
                          rnn_dropout=0.5, hidden_size=32, cell="biGRU",
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

# In kosshre chun overfit shode
# study 1
# accuracy f1 kappa precision recall
# Mean  0.958087  0.804051  0.781523   0.680086  0.985083

# annotation:
# accuracy f1 kappa precision recall
# Mean  0.871081  0.558240  0.484313   0.641480  0.497457


# demo:
# 7 epochs, 10 folds, learnin_rate = .0005, hidden = 64, drop_out = .5
# accuracy f1 kappa precision recall
#      accuracy        f1     kappa  precision    recall
#8
#0     0.870482  0.574257  0.500420   0.690476  0.491525
#1     0.889831  0.606061  0.546720   0.810811  0.483871
#2     0.881443  0.596491  0.528230   0.680000  0.531250
#3     0.920424  0.651163  0.608461   0.823529  0.538462
#4     0.898123  0.693548  0.632474   0.704918  0.682540
#5     0.859599  0.505051  0.433648   0.757576  0.378788
#6     0.897281  0.660000  0.601318   0.785714  0.568966
#7     0.883289  0.560000  0.495222   0.682927  0.474576
#8     0.880759  0.551020  0.483095   0.613636  0.500000
#9     0.855586  0.522523  0.441015   0.644444  0.439394
#Mean  0.883682  0.592011  0.527060   0.719403  0.508937


