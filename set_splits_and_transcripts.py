'''
J. Elizabeth Liebl; jliebl@gmu.edu
03/09/2026
ASR for ICLDC
Set Train/Eval Splits; Collect Transcript Characters
'''

import pandas as pd

train = ["ab", "ba", "bas", "bg", "ca", "cs", "cv", "et", "ha", "kmr", "ky", "lt", "luo",
        "nl", "pl", "ro", "ru", "rw", "sk", "sl", "sv-SE", "ta", "ug", "uk", "uz", "yo"]

test = ["hsb", "lg", "tt"]

seed = 42
char_set_list = []

for t in train:
    full_df = pd.read_csv(f"/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/full/{t}_data.tsv", sep="\t")
    samp_df = full_df.sample(n=2400, random_state=seed)
    train_df = samp_df.iloc[:2000, :]
    test_df = samp_df.iloc[2000:, :]
    train_df.to_csv(f"/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/train/{t}_train.tsv", sep="\t", index=False)
    test_df.to_csv(f"/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/test/{t}_test.tsv", sep="\t", index=False)
    train_trans = list(train_df["Transcript"])
    test_trans = list(test_df["Transcript"])
    all_trans = train_trans + test_trans
    unique = set("".join(all_trans))
    char_set_list.append(unique)

for t in test:
    full_df = pd.read_csv(f"/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/full/{t}_data.tsv", sep="\t")
    samp_df = full_df.sample(n=400, random_state=seed)
    samp_df.to_csv(f"/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/test/{t}_test.tsv", sep="\t", index=False)
    all_trans = list(samp_df["Transcript"])
    unique = set("".join(all_trans))
    char_set_list.append(unique)

char_set = set().union(*char_set_list)
with open("/scratch/jliebl/ASR_ComputEL/char_set.txt", "w") as output:
    output.write("\n".join(char_set) + "\n")