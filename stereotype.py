import pandas as pd
import json
import gensim
import numpy as np
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import pickle

def create_dictionary():
    groups = json.load(open("Data/stereo.json", "r"))

    SGTs = [x.replace("\n", "") for x in
            open("../HateSpeech/extended_SGT.txt", "r").readlines()]



    glove_file = datapath("/home/aida/Data/word_embeddings/GloVe/glove.840B.300d.txt")
    tmp_file = get_tmpfile("glove.txt")
    _ = glove2word2vec(glove_file, tmp_file)
    model = KeyedVectors.load_word2vec_format(tmp_file)

    vecs = {"SGT": dict(),
            "agency": dict(),
            "communion": dict()}

    not_found = {"agency": list(),
                 "communion": list()}
    for group in ["agency", "communion"]:
        words = [x.replace("*", "") for x in groups[group]]
        for word in words:
            try:
                vecs[group][word] = model[word]
            except Exception:
                not_found[group].append(word)


    for voc in model.wv.vocab:
        for group in groups:
            for word in not_found[group]:
                if voc.startswith(word):
                    vecs[group][word] = model[voc]
                    break

    for s in SGTs:
        try:
            vecs["SGT"][s] = model[s]
        except Exception:
            continue

    pickle.dump(vecs, open("Data/dictionary.pkl", "wb"))

def euclidean(x, y):
    #return np.linalg.norm(x - y)
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def differences():
    groups = ["agency", "communion"]
    vecs = pickle.load(open("Data/dictionary.pkl", "rb"))
    distances = {"sgt": list(),
                 "agency": list(),
                 "communion": list()}
    for s in vecs["SGT"]:
        distances["sgt"].append(s)
        for group in groups:
            dis = [euclidean(vecs["SGT"][s], vecs[group][x]) for x in vecs[group].keys()]
            distances[group].append(sum(dis) / len(dis))
            #avg = sum(list(vecs[group].values())) / len(vecs[group].values())
            #distances[group].append(euclidean(avg, vecs["SGT"][s]))
    #pd.DataFrame.from_dict(distances).to_csv("Data/sgt_stereotypes.csv", index=False)
    pd.DataFrame.from_dict(distances).to_csv("Data/sgt_stereotypes.csv", index=False)

def join_sgt():
    #stereo = pd.read_csv("Data/sgt_stereotypes.csv")
    stereo = pd.read_csv("Data/sgt_stereotypes.csv")
    fp = pd.read_csv("../HateSpeech/biased/fp_SGT.csv")
    fp.merge(stereo, right_on="sgt", left_on="Change")[["sgt", "Frequency", "agency", "communion"]]\
        .to_csv("Data/study1_ddr.csv", index=False)

def visualize():
    change = pd.read_csv("../HateSpeech/biased/fp_change.csv")
    sgt = pd.read_csv("Data/study1_norm.csv")

    dots = {x: {"agency": 0, "communion": 0} for x in sgt["sgt"]}
    for i, row in sgt.iterrows():
        dots[row["sgt"]]["agency"] = row["agency"]
        dots[row["sgt"]]["communion"] = row["communion"]

    lines = {"a1": list(),
             "c1": list(),
             "a2": list(),
             "c2": list(),
             "Freuency": list()}

    for i, row in change.iterrows():
        lines["a1"].append(dots[row["From"]]["agency"])
        lines["c1"].append(dots[row["From"]]["communion"])
        lines["a2"].append(dots[row["To"]]["agency"])
        lines["c2"].append(dots[row["To"]]["communion"])
        lines["Freuency"].append(row["Frequency"])

    pd.DataFrame.from_dict(lines).to_csv("Data/study1_changes.csv", index=False)

if __name__ == "__main__":
    create_dictionary()
    differences()
    join_sgt()
    #visualize()


