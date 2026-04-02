'''
J. Elizabeth Liebl; jliebl@gmu.edu
03/10/2026
ASR for ComputEL
Create Datasets
'''

import pandas as pd
from datasets import Dataset
from datasets import load_from_disk

def create_dataset(model_name, train_eval, concat_df):
    audios = concat_df["Audio"]
    sents = concat_df["Transcript"]
    df1_cols = {"audio":audios, "sentence":sents}
    df1 = pd.DataFrame(df1_cols)
    df1 = df1.sample(frac=1).reset_index(drop=True)
    df1_dataset = Dataset.from_pandas(df1)
    df1_dataset.save_to_disk(f"/scratch/jliebl/ASR_ComputEL/Datasets/{model_name}_{train_eval}")
    return print(f"{model_name} {train_eval} dataset created and saved successfully!")

# Random
# Train
sv_tr = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/train/sv-SE_train.tsv", sep="\t")
uk_tr = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/train/uk_train.tsv", sep="\t")
ab_tr = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/train/ab_train.tsv", sep="\t")
ro_tr = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/train/ro_train.tsv", sep="\t")

rand_tr_extra = pd.concat([sv_tr, uk_tr, ab_tr, ro_tr])
create_dataset("rand", "tr", rand_tr_extra)

# eval
sv_te = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/eval/sv-SE_eval.tsv", sep="\t")
uk_te = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/eval/uk_eval.tsv", sep="\t")
ab_te = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/eval/ab_eval.tsv", sep="\t")
ro_te = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/eval/ro_eval.tsv", sep="\t")

rand_te_extra = pd.concat([sv_te, uk_te, ab_te, ro_te])
create_dataset("rand", "te", rand_te_extra)

# hsb_geo
# Train
cs_tr = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/train/cs_train.tsv", sep="\t")
pl_tr = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/train/pl_train.tsv", sep="\t")
sk_tr = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/train/sk_train.tsv", sep="\t")
sl_tr = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/train/sl_train.tsv", sep="\t") 

hsb_geo_tr = pd.concat([cs_tr, pl_tr, sk_tr, sl_tr])
create_dataset("hsb_geo", "tr", hsb_geo_tr)

# eval
cs_te = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/eval/cs_eval.tsv", sep="\t")
pl_te = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/eval/pl_eval.tsv", sep="\t")
sk_te = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/eval/sk_eval.tsv", sep="\t")
sl_te = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/eval/sl_eval.tsv", sep="\t")

hsb_geo_te = pd.concat([cs_te, pl_te, sk_te, sl_te])
create_dataset("hsb_geo", "te", hsb_geo_te)

# hsb fam
# Train
bg_tr = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/train/bg_train.tsv", sep="\t")

hsb_fam_tr = pd.concat([cs_tr, pl_tr, sk_tr, bg_tr])
create_dataset("hsb_fam", "tr", hsb_fam_tr)

# eval
bg_te = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/eval/bg_eval.tsv", sep="\t")

hsb_fam_te = pd.concat([cs_te, pl_te, sk_te, bg_te])
create_dataset("hsb_fam", "te", hsb_fam_te)

# hsb phon
# Train
lt_tr = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/train/lt_train.tsv", sep="\t")

hsb_phon_tr = pd.concat([bg_tr, lt_tr, ro_tr, uk_tr])
create_dataset("hsb_phon", "tr", hsb_phon_tr)

# eval
lt_te = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/eval/lt_eval.tsv", sep="\t")

hsb_phon_te = pd.concat([bg_te, lt_te, ro_te, uk_te])
create_dataset("hsb_phon", "te", hsb_phon_te)

# lg geo
# Train
luo_tr = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/train/luo_train.tsv", sep="\t")
rw_tr = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/train/rw_train.tsv", sep="\t")
sw_tr = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/train/sw_train.tsv", sep="\t")
bas_tr = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/train/bas_train.tsv", sep="\t")

lg_geo_tr = pd.concat([luo_tr, rw_tr, sw_tr, bas_tr])
create_dataset("lg_geo", "tr", lg_geo_tr)

# eval
luo_te = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/eval/luo_eval.tsv", sep="\t")
rw_te = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/eval/rw_eval.tsv", sep="\t")
sw_te = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/eval/sw_eval.tsv", sep="\t") 
bas_te = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/eval/bas_eval.tsv", sep="\t")

lg_geo_te = pd.concat([luo_te, rw_te, sw_te, bas_te])
create_dataset("lg_geo", "te", lg_geo_te)

# lg fam
# Train
yo_tr = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/train/yo_train.tsv", sep="\t")

lg_fam_tr = pd.concat([rw_tr, sw_tr, bas_tr, yo_tr])
create_dataset("lg_fam", "tr", lg_fam_tr)

# eval
yo_te = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/eval/luo_eval.tsv", sep="\t")

lg_fam_te = pd.concat([rw_te, sw_te, bas_te, yo_te])
create_dataset("lg_fam", "te", lg_fam_te)

# lg phon
# Train
nl_tr = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/train/nl_train.tsv", sep="\t")
id_tr = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/train/id_train.tsv", sep="\t")

lg_phon_tr = pd.concat([luo_tr, nl_tr, id_tr, bas_tr])
create_dataset("lg_phon", "tr", lg_phon_tr)

# eval
nl_te = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/eval/nl_eval.tsv", sep="\t")
id_te = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/eval/id_eval.tsv", sep="\t")


lg_phon_te = pd.concat([luo_te, nl_te, id_te, bas_te])
create_dataset("lg_phon", "te", lg_phon_te)

# tt geo
# Train
cv_tr = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/train/cv_train.tsv", sep="\t")
ru_tr = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/train/ru_train.tsv", sep="\t")
ba_tr = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/train/ba_train.tsv", sep="\t")
et_tr = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/train/et_train.tsv", sep="\t")

tt_geo_tr = pd.concat([cv_tr, ru_tr, ba_tr, et_tr])
create_dataset("tt_geo", "tr", tt_geo_tr)

# eval
cv_te = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/eval/cv_eval.tsv", sep="\t")
ru_te = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/eval/ru_eval.tsv", sep="\t")
ba_te = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/eval/ba_eval.tsv", sep="\t")
et_te = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/eval/et_eval.tsv", sep="\t")

tt_geo_te = pd.concat([cv_te, ru_te, ba_te, et_te])
create_dataset("tt_geo", "te", tt_geo_te)

# tt fam
# Train
ky_tr = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/train/ky_train.tsv", sep="\t")
ug_tr = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/train/ug_train.tsv", sep="\t")
uz_tr = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/train/uz_train.tsv", sep="\t")

tt_fam_tr = pd.concat([ba_tr, ky_tr, ug_tr, uz_tr])
create_dataset("tt_fam", "tr", tt_fam_tr)

# eval
ky_te = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/eval/ky_eval.tsv", sep="\t")
ug_te = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/eval/ug_eval.tsv", sep="\t")
uz_te = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/eval/uz_eval.tsv", sep="\t")

tt_fam_te = pd.concat([ba_te, ky_te, ug_te, uz_te])
create_dataset("tt_fam", "te", tt_fam_te)

# tt phon
# Train
hi_tr = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/train/hi_train.tsv", sep="\t")

tt_phon_tr = pd.concat([ba_tr, hi_tr, ug_tr, pl_tr])
create_dataset("tt_phon", "tr", tt_phon_tr)

# eval
hi_te = pd.read_csv("/scratch/jliebl/ASR_ComputEL/CV_VC_Data/Dataframes/eval/hi_eval.tsv", sep="\t")


tt_phon_te = pd.concat([ba_te, hi_te, ug_te, pl_te])
create_dataset("tt_phon", "te", tt_phon_te)

print("All dataset creation complete!")
