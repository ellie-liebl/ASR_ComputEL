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

def eval_model(model, training_langs):
    MODEL_DIR = f"/scratch/jliebl/ASR_ComputEL/ASR_ComputEL/Models/{model}"

    device = 0   # GPU 0 on Hopper
    transcriber = pipeline(
        task="automatic-speech-recognition",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        feature_extractor=MODEL_DIR,
        device=device
    )
    for l in training_langs:
        eval_df = pd.read_csv(f"/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/test/{l}_test.tsv", sep="\t")

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
        eval_df.to_csv(f"/scratch/jliebl/ASR_ComputEL/ASR_ComputEL/Results/{model}_{l}_seen_results.tsv", sep="\t", index=False)

hsb_fam_langs = ["cs", "pl", "sk", "bg"]
hsb_geo_langs = ["cs", "pl", "sk", "sl"]
hsb_phon_langs = ["bg", "lt", "ro", "uk"]
lg_fam_langs = ["rw", "sw", "bas", "yo"]
lg_geo_langs = ["luo", "rw", "sw", "bas"]
lg_phon_langs = ["luo", "nl", "id", "bas"]
tt_fam_langs = ["ba", "ky", "ug", "uz"]
tt_geo_langs = ["cv", "ru", "ba", "et"] 
tt_phon_langs = ["ba", "hi", "ug", "pl"]
rand_langs = ["sv-SE", "uk", "ab", "ro"]

# eval_model("hsb_fam", hsb_fam_langs)
# eval_model("hsb_geo", hsb_geo_langs)
# eval_model("hsb_phon", hsb_phon_langs)
# eval_model("lg_fam", lg_fam_langs)
# eval_model("lg_geo", lg_geo_langs)
# eval_model("lg_phon", lg_phon_langs)
# eval_model("tt_fam", tt_fam_langs)
# eval_model("tt_geo", tt_geo_langs)
# eval_model("tt_phon", tt_phon_langs)
eval_model("rand", rand_langs)