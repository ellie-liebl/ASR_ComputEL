'''
J. Elizabeth Liebl; jliebl@gmu.edu
03/10/2026
ASR for ComputEL
Preprocess Transcripts
'''

import pandas as pd
import re
from tqdm import tqdm

def clean_transcripts(transcript_list):
    clean_transcripts = []
    for t in tqdm(transcript_list):
        x = re.sub("t͡s", "ʦ", t)
        y = re.sub("d͡z", "ʣ", x)
        z = re.sub("t͡ʃ", "ʧ", y)
        a = re.sub("d͡ʒ", "ʤ", z)
        b = re.sub("t͡ɕ", "ʨ", a)
        c = re.sub("d͡ʑ", "ʥ", b)
        d = re.sub("d͡ʐ", "ꭦ", c)
        e = re.sub("t͡ʂ", "ꭧ", d)
        f = re.sub("̇", "", e)
        g = re.sub("̩", "", f)
        h = re.sub("ˈ", "", g)
        i = re.sub("ʼ", "", h)
        j = re.sub("̈̈", "", i)
        k = re.sub("̪", "", j)
        m = re.sub("́", "", k)
        n = re.sub("ː", "", m)
        o = re.sub("̀", "", n)
        p = re.sub("ˑ", "", o)
        q = re.sub("̃", "", p)
        r = re.sub("̂", "", q)
        s = re.sub("̯", "", r)
        tt = re.sub("̝", "", s)
        u = re.sub("̠", "", tt)
        v = re.sub("̌", "", u)
        w = re.sub("ʻ", "", v)
        cleaner = re.sub("ˌ", "", w)
        oops = re.sub("͡", "", cleaner)
        clean = re.sub("spn", "*", oops)
        clean_transcripts.append(clean)
    return clean_transcripts


train = ["ab", "ba", "bas", "bg", "cs", "cv", "et", "hi", "id", "ky", "lt", "luo",
        "nl", "pl", "ro", "ru", "rw", "sk", "sl", "sv-SE", "sw", "ug", "uk", "uz", 
        "yo"]

test = ["hsb", "lg", "tt", "ab", "ba", "bas", "bg", "cs", "cv", "et", "hi", "id", 
        "ky", "lt", "luo", "nl", "pl", "ro", "ru", "rw", "sk", "sl", "sv-SE", "sw", 
        "ug", "uk", "uz", "yo"]

for l in tqdm(train):
    train_df = pd.read_csv(f"/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/train/{l}_train.tsv", sep="\t")
    trans = list(train_df["Transcript"])
    train_df.rename(columns={"Transcript":"Unprocessed Transcript"}, inplace=True)
    clean = clean_transcripts(trans)
    train_df["Transcript"] = clean
    train_df.to_csv(f"/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/train/{l}_train.tsv", sep="\t", index=False)
    eval_df = pd.read_csv(f"/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/eval/{l}_eval.tsv", sep="\t")
    trans = list(eval_df["Transcript"])
    eval_df.rename(columns={"Transcript":"Unprocessed Transcript"}, inplace=True)
    clean = clean_transcripts(trans)
    eval_df["Transcript"] = clean
    eval_df.to_csv(f"/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/eval/{l}_eval.tsv", sep="\t", index=False)


for l in tqdm(test):
    test_df = pd.read_csv(f"/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/test/{l}_test.tsv", sep="\t")
    trans = list(test_df["Transcript"])
    test_df.rename(columns={"Transcript":"Unprocessed Transcript"}, inplace=True)
    clean = clean_transcripts(trans)
    test_df["Transcript"] = clean
    test_df.to_csv(f"/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/test/{l}_test.tsv", sep="\t", index=False)