import re
import pandas as pd
from collections import Counter

def wonky_parser(fn, cols):
    txt = open(fn).read()
    #                          This is where I specified 12 tabs
    #                                       V
    preparse = re.findall('(([^\t]*\t[^\t]*){' + str(cols) + '}(\n|\Z))', txt)
    parsed = [t[0].replace("\n", "").split('\t') for t in preparse]
    return pd.DataFrame(parsed[1:], columns=parsed[0])

def read_raw():
    annotation_files = ["/home/aida/Data/Gab/20k/Annotated/Hate_" +  str(i) + ".tsv"
                        for i in range(5)]
    annotation_files.append("/home/aida/Data/Gab/7k/Annotated/Hate_5.tsv" )

    columns = ['Tweet ID', 'Username', 'Foundation', 'Text']

    for file in annotation_files:
        print(file)
        raw = wonky_parser(file, 8 if "5" in file else 12)
        uniq = raw.drop_duplicates(subset=["Tweet ID", "Username"], keep="last")
        annotations = uniq[columns]
        annotations.to_csv("Annotations.csv", index=False, header=False, mode="a")

def get_label():
    df = pd.read_csv("annotations.csv")
    docs = set(df["Tweet ID"])
    annotations = {doc: {"annotations": dict(),
                         "label": ""} for doc in docs}
    for i, row in df.iterrows():
        annotations[row["Tweet ID"]]["annotations"][row["Username"]] = \
            row["Foundation"].split(",")

    for doc in docs:
        votes = Counter([a for an in annotations[doc]["annotations"].values()
                         for a in an])
        annotations[doc]["label"] = [vote for vote in votes.keys() if votes[vote] >
                                     len(annotations[doc]["annotations"].keys()) / 2]

    final = list()
    for i, row in df.iterrows():
        labels = annotations[row["Tweet ID"]]["label"]
        final.append(",".join(l for l in labels))

    df["Labels"] = pd.Series(final)
    df.to_csv("annotations_maj.csv", index=False)

def get_hate():
    df = pd.read_csv("annotations_maj.csv")
    hate, maj_hate = list(), list()
    for i, row in df.iterrows():
        if isinstance(row["Foundation"], str) and \
                ("cv" in row["Foundation"] or "hd" in row["Foundation"]):
            hate.append(1)
        else:
            if "nh" not in row["Foundation"]:
                print(row["Foundation"])
            hate.append(0)

        if isinstance(row["Labels"], str) and \
                ("cv" in row["Labels"] or "hd" in row["Labels"]):
            maj_hate.append(1)
        else:
            maj_hate.append(0)
    df["Hate"] = pd.Series(hate)
    df["Maj_Hate"] = pd.Series(maj_hate)
    df.to_csv("annotations_maj.csv", index=False)

def aggregate():
    df = pd.read_csv("annotations_maj.csv")
    docs = set(df["Tweet ID"])
    annotators = set(df["Username"])
    annotations = {doc: dict() for doc in docs}

    for i, row in df.iterrows():
        annotations[row["Tweet ID"]][row["Username"]] = row["Hate"]
        annotations[row["Tweet ID"]]["text"] = row["Text"]

    anno_df = {i: list() for i, user in enumerate(annotators)}
    anno_df["id"] = list()
    anno_df["text"] = list()

    for key, val in annotations.items():
        anno_df["id"].append(key)
        anno_df["text"].append(val["text"])
        for annotator, i in enumerate(annotators):
            if annotator in val.keys():
                anno_df[annotator].append(val[annotator])
            else:
                anno_df[annotator].append(2)

    pd.DataFrame.from_dict(anno_df).to_csv("posts.csv", index=False)

if __name__ == "__main__":
    #read_raw()
    #get_labe()
    #get_hate()
    aggregate()
