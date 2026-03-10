'''
J. Elizabeth Liebl; jliebl@gmu.edu
3/10/2026
ASR for ComputEL
Evaluate Models
'''
# Import Tools
from transformers import pipeline
import torch
import regex as re
import pandas as pd
from jiwer import cer

hsb_models = ["hsb_fam", "hsb_geo", "hsb_phon", "rand"]
lg_models = ["lg_fam", "lg_geo", "lg_phon", "rand"]
tt_models = ["tt_fam", "tt_geo", "tt_phon", "rand"]

for m in hsb_models:
    MODEL_DIR = f"/scratch/jliebl/ASR_ComputEL/Models/{m}"

    device = 0   # GPU 0 on Hopper
    transcriber = pipeline(
        task="automatic-speech-recognition",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        feature_extractor=MODEL_DIR,
        device=device
    )

    eval_df = pd.read_pickle("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/test/hsb_test.tsv")

    preds =[]
    golds = []
    cers = []

    for idx, row in eval_df.iterrows():
        pred = transcriber(row["Audio"])
        pred_text = pred["text"]
        preds.append(pred_text)
        gold = row["Transcript"]
        golds.append(gold)
        error = cer(gold, pred_text)
        cers.append(error)
        print(f"Row {idx} complete.")

    eval_df["Prediction"] = preds
    eval_df["CER scores"] = cers
    eval_df.to_csv("/scratch/jliebl/ASR_ComputEL/Results/{m}_eval_results_hsb.tsv", sep="\t", index=False)

for m in lg_models:
    MODEL_DIR = f"/scratch/jliebl/ASR_ComputEL/Models/{m}"

    device = 0   # GPU 0 on Hopper
    transcriber = pipeline(
        task="automatic-speech-recognition",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        feature_extractor=MODEL_DIR,
        device=device
    )

    eval_df = pd.read_pickle("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/test/lg_test.tsv")

    preds =[]
    golds = []
    cers = []

    for idx, row in eval_df.iterrows():
        pred = transcriber(row["Audio"])
        pred_text = pred["text"]
        preds.append(pred_text)
        gold = row["Transcript"]
        golds.append(gold)
        error = cer(gold, pred_text)
        cers.append(error)
        print(f"Row {idx} complete.")

    eval_df["Prediction"] = preds
    eval_df["CER scores"] = cers
    eval_df.to_csv("/scratch/jliebl/ASR_ComputEL/Results/{m}_eval_results_lg.tsv", sep="\t", index=False)

for m in tt_models:
    MODEL_DIR = f"/scratch/jliebl/ASR_ComputEL/Models/{m}"

    device = 0   # GPU 0 on Hopper
    transcriber = pipeline(
        task="automatic-speech-recognition",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        feature_extractor=MODEL_DIR,
        device=device
    )

    eval_df = pd.read_pickle("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/test/tt_test.tsv")

    preds =[]
    golds = []
    cers = []

    for idx, row in eval_df.iterrows():
        pred = transcriber(row["Audio"])
        pred_text = pred["text"]
        preds.append(pred_text)
        gold = row["Transcript"]
        golds.append(gold)
        error = cer(gold, pred_text)
        cers.append(error)
        print(f"Row {idx} complete.")

    eval_df["Prediction"] = preds
    eval_df["CER scores"] = cers
    eval_df.to_csv("/scratch/jliebl/ASR_ComputEL/Results/{m}_eval_results_tt.tsv", sep="\t", index=False)

