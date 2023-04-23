import os
import sys
import json
import re
import nltk

import pandas as pd

ENG_TO_SLO_TITLE = {
    "THE GOLDEN BIRD" : "zlata ptica",
    "JORINDA AND JORINDEL" : "jorinda in joringel",
    "THE STRAW, THE COAL, AND THE BEAN" : "slamica ogeljcek in fizolcek",
    "THE FROG-PRINCE" : "zabji princ",
    "THE ADVENTURES OF CHANTICLEER AND PARTLET" : "boter petelin in njegova zgodba",
    "HANSEL AND GRETEL" : "janko in metka",
    "THE MOUSE, THE BIRD, AND THE SAUSAGE" : "misek pticka in klobasa",
    "LITTLE RED-CAP [LITTLE RED RIDING HOOD]" : "rdeca kapica",
    "RUMPELSTILTSKIN" : "spicparkeljc",
    "ASHPUTTEL" : "pepelka",
    "THE WOLF AND THE SEVEN LITTLE KIDS" : "volk in sedem kozlickov",
    "THE FOX AND THE CAT" : "lisica in macka",
    "THE GOLDEN GOOSE" : "zlata gos",
    "THE SEVEN RAVENS" : "sedem krokarjev",
    "THE STORY OF THE YOUTH WHO WENT FORTH TO LEARN WHAT FEAR WAS" : "o mladenicu ki bi rad strah poznal"
}


if __name__ == "__main__":

    eng_df = pd.read_csv("../data/eng_raw/grimms_fairytales.csv", index_col=0)
    eng_df = eng_df.rename(columns={"Title": "eng_title", "Text": "eng_text"})
    eng_df["eng_text_len"] = eng_df["eng_text"].apply(lambda x: len(re.findall(r'\w+', x)))

    slo_dir = "../data/slo_raw/txt"
    all_slo_titles, all_slo_texts = [], []
    for fn in os.listdir(slo_dir):
        title = fn.rsplit(".")[0].replace("-", " ")
        all_slo_titles.append(title)
        with open(os.path.join(slo_dir, fn), "r", encoding="utf-8") as f:
            all_slo_texts.append(f.read()) 
    slo_df = pd.DataFrame(data={"slo_title": all_slo_titles, "slo_text": all_slo_texts})
    slo_df["slo_text_len"] = slo_df["slo_text"].apply(lambda x: len(re.findall(r'\w+', x)))

    df = eng_df[eng_df["eng_title"].isin(ENG_TO_SLO_TITLE)]
    mp = lambda t: ENG_TO_SLO_TITLE[t]
    df["slo_title"] = df["eng_title"].apply(mp)
    df = df.merge(slo_df, right_on="slo_title", left_on="slo_title", how="inner")

    df.to_csv("../data/all_data.csv")

    # Create split files for labels
    for i, row in df.iterrows():
        # slovene
        title = row["slo_title"].replace(" ", "_")
        text = row["slo_text"]
        slo_sentences = nltk.sent_tokenize(text, language="slovene")
        labels = [1] * len(slo_sentences)
        slo_df = pd.DataFrame(data={"label": labels, "sentence": slo_sentences})
        slo_df.to_csv(f"../data/slo_processed/{title}.csv")

        # english
        title = row["eng_title"].replace(" ", "_")
        text = row["eng_text"]
        eng_sentences = nltk.sent_tokenize(text, language="english")
        labels = [1] * len(eng_sentences)
        eng_df = pd.DataFrame(data={"label": labels, "sentence": eng_sentences})
        eng_df.to_csv(f"../data/eng_processed/{title}.csv")






