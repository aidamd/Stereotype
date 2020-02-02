import re
import pandas as pd
from collections import Counter
import json

"""
def wonky_parser(fn, cols):
    txt = open(fn).read()
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
"""

def get_label(data_path):
    """
    df = pd.read_csv("annotations.csv")
    docs = set(df["Tweet ID"])
    annotations = {doc: {"annotations": dict(),
                         "label": ""} for doc in docs}
    """
    annotations = json.load(open(data_path, "r"))

    df = {"Tweet ID": list(),
          "Text": list(),
          "Username": list(),
          "Foundation": list()}
    for annotation in annotations:
        if len(annotation["annotations"]) > 2:
            for anno in annotation["annotations"]:
                df["Text"].append(annotation["tweet_text"])
                df["Tweet ID"].append(annotation["tweet_id"])
                df["Username"].append(anno["annotator"])
                df["Foundation"].append(anno["annotation"])
    pd.DataFrame.from_dict(df).to_csv("Data/annotations.csv", index= False)


def get_hate():
    df = pd.read_csv("Data/annotations.csv")
    hate, maj_hate = list(), list()
    drop = list()
    for i, row in df.iterrows():
        if row["Username"] == "Praveen":
            drop.append(i)
        if isinstance(row["Foundation"], str) and \
                ("cv" in row["Foundation"] or "hd" in row["Foundation"]):
            hate.append(1)
        else:
            #if "nh" not in row["Foundation"]:
            #    print(row["Foundation"])
            hate.append(0)
        """
        if isinstance(row["Labels"], str) and \
                ("cv" in row["Labels"] or "hd" in row["Labels"]):
            maj_hate.append(1)
        else:
            maj_hate.append(0)
        """
    df["Hate"] = pd.Series(hate)
    #df["Maj_Hate"] = pd.Series(maj_hate)
    df = df.drop(drop)
    df.to_csv("Data/annotations_maj.csv", index=False)

def aggregate():
    df = pd.read_csv("Data/annotations_maj.csv")
    docs = set(df["tweet_id"])
    annotators = list(set(df["username"]))
    annotators.sort()
    annotations = {doc: dict() for doc in docs}

    for i, row in df.iterrows():
        annotations[row["tweet_id"]][row["username"]] = row["hate"]
        annotations[row["tweet_id"]]["text"] = row["text"]

    anno_df = {i: list() for i, user in enumerate(annotators)}
    anno_df["id"] = list()
    anno_df["text"] = list()
    anno_df["hate"] = list()
    anno_df["agreement"] = list()

    for key, val in annotations.items():
        anno_df["id"].append(key)
        anno_df["text"].append(val["text"])
        hate = {0: 0, 1: 0}
        for i, annotator in enumerate(annotators):
            if annotator in val.keys():
                anno_df[i].append(val[annotator])
                hate[val[annotator]] += 1
            else:
                anno_df[i].append(2)
        anno_df["hate"].append(1 if hate[1] > hate[0] else 0)
        agree = max(hate[0] / (hate[0] + hate[1]),
                    hate[1] / (hate[0] + hate[1]))
        if agree < 0.67:
            anno_df["agreement"].append(0)
        elif agree < 0.84:
            anno_df["agreement"].append(1)
        else:
            anno_df["agreement"].append(2)

    annotator_dict = {annotator: i for i, annotator in enumerate(annotators)}
    print(annotator_dict)
    for i, row in df.iterrows():
        df.at[i, "username"] = annotator_dict[row["username"]]
    json.dump(annotator_dict, open("annotators.json", "w"))
    df.to_csv("Data/annotations_id.csv", index=False)

    posts = pd.DataFrame.from_dict(anno_df)
    posts.to_csv("Data/posts.csv", index=False)

def iat():
    df = pd.read_csv("Data/demo.csv")
    anno = json.load(open("annotators.json", "r"))
    df = df.dropna(subset=["Username"])
    drop = list()
    for i, row in df.iterrows():
        try:
            df.at[i, "Username"] = anno[row["Username"]]
        except Exception:
            drop.append(i)
    df = df.drop(drop)
    cols = ["Username", "Race", "Gender-Career", "Sexuality", "Religion",
        "negative_belief", "offender_punishment", "deterrence", "victim_harm"]
    iat = pd.DataFrame()
    cols.remove("Username")
    for col in cols:
        mean = (df[col].max + df[col].min()) / 2
        iat[col] = (df[col] - mean) / (df[col].max() - df[col].min())
    iat["Username"] = df["Username"]
    iat.to_csv("Data/demo_clean.csv", index=False)

def annotattor_demo():
    df = pd.read_csv("Data/Annotators_demo.csv")
    anno = {"Annotator ID": list(),
            "negative_belief": list(),
            "offender_punishment": list(),
            "deterrence": list(),
            "victim_harm": list(),
            "political_view": list()}

    for i, row in df.iterrows():
        #if isinstance(row["Q17"], str):
        try:
            if int(row["Q17"]) > 0:
                anno["Annotator ID"].append(int(row["Q17"]))
            else:
                continue
        except Exception:
            continue

        b, o, d, v = 0, 0, 0, 0
        b_count, o_count, d_count, v_count = 0, 0, 0, 0

        for x in range(1, 17):
            col = "Q89_" + str(x)
            if isinstance(row[col], str):
                b_count += 1
                if x not in [13, 15, 16]:
                    b += int(row[col])
                else:
                    b += (7 - int(row[col]))
        anno["negative_belief"].append(b / b_count)

        for x in range(17, 21):
            col = "Q89_" + str(x)
            if isinstance(row[col], str):
                o_count += 1
                o += int(row[col])
        anno["offender_punishment"].append(o / o_count)

        for x in range(21, 24):
            col = "Q89_" + str(x)
            if isinstance(row[col], str):
                d_count += 1
                d += int(row[col])
        anno["deterrence"].append(d / d_count)

        for x in range(24, 28):
            col = "Q89_" + str(x)
            if isinstance(row[col], str):
                v_count += 1
                v += int(row[col])
        anno["victim_harm"].append(v / v_count)
        anno["political_view"].append((int(row["Q24"]) + int(row["Q22.1"])) / 2)
    anno = pd.DataFrame.from_dict(anno)
    anno.to_csv("Data/annotators_hate.csv", index = False)
    iat = pd.read_csv("Data/IAT.csv")
    iat.merge(anno, on="Annotator ID").to_csv("Data/demo.csv", index=False)

def stereotype():
    keys = ["agency", "communion"]
    str_dict = {k: list() for k in keys}
    for k in keys:
        lines = open("Data/" + k + "_temp.txt", "r").readlines()
        words = [word.replace("\n", "") for word in lines if word != "\n"]
        str_dict[k] = words
    json.dump(str_dict, open("Data/stereo.json", "w"), indent=4)


if __name__ == "__main__":
    #get_label("/home/aida/Data/Gab/full_disaggregated.json")
    #get_hate()
    #aggregate()
    iat()
    #annotattor_demo()
    #stereotype()