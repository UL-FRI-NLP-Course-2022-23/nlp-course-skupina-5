import sys
import os
import pandas as pd
import classla

if __name__ == "__main__":
    df = pd.read_csv("../data/data.csv")

    classla.download("sl")
    nlp_sl = classla.Pipeline("sl", processors="tokenize")

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
