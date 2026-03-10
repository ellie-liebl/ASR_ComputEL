'''
J. Elizabeth Liebl; jliebl@gmu.edu
11/28/2025
LING 731 - Phonetics
Project: Eval MIPA model
'''
# Import Tools
from transformers import pipeline
import torch
import regex as re
import pandas as pd
from jiwer import cer

MODEL_DIR = "/scratch/jliebl/Phonetics/out"

device = 0   # GPU 0 on Hopper
transcriber = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR,
    feature_extractor=MODEL_DIR,
    device=device
)

eval_df = pd.read_pickle("/scratch/jliebl/Phonetics/mipa_eval_200_cluster.pkl")

preds =[]
golds = []
cers = []

for idx, row in eval_df.iterrows():
    pred = transcriber(row["Audio Path"])
    pred_text = pred["text"]
    preds.append(pred_text)
    gold = row["Transcript"]
    golds.append(gold)
    error = cer(gold, pred_text)
    cers.append(error)
    print(f"Row {idx} complete.")

eval_df["Prediction"] = preds
eval_df["CER scores"] = cers
eval_df.to_pickle("/scratch/jliebl/Phonetics/mipa_eval_200_cluster_with_predictons.pkl")
eval_df.to_csv("/scratch/jliebl/Phonetics/mipa_eval_200_cluster_with_predictons.tsv", sep="\t", index=False)

