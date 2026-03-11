'''
J. Elizabeth Liebl; jliebl@gmu.edu
03/09/2026
ASR for ICLDC
Gen Transcript-Audio file dataset
'''

import pandas as pd
import numpy as np

hsb_fam = pd.read_csv("/scratch/jliebl/ASR_ComputEL/ASR_ComputEL/Results/hsb_fam_eval_results_hsb.tsv", sep="\t")
hf_cers = hsb_fam["CER scores"]
avg_hf_cer = np.mean(hf_cers)
print(f"Average CER for hsb_fam is {avg_hf_cer:.2f}")

hsb_geo = pd.read_csv("/scratch/jliebl/ASR_ComputEL/ASR_ComputEL/Results/hsb_geo_eval_results_hsb.tsv", sep="\t")
hg_cers = hsb_geo["CER scores"]
avg_hg_cer = np.mean(hg_cers)
print(f"Average CER for hsb_geo is {avg_hg_cer:.2f}")

hsb_phon = pd.read_csv("/scratch/jliebl/ASR_ComputEL/ASR_ComputEL/Results/hsb_phon_eval_results_hsb.tsv", sep="\t")
hp_cers = hsb_phon["CER scores"]
avg_hp_cer = np.mean(hp_cers)
print(f"Average CER for hsb_phon is {avg_hp_cer:.2f}")

hsb_rand = pd.read_csv("/scratch/jliebl/ASR_ComputEL/ASR_ComputEL/Results/rand_eval_results_hsb.tsv", sep="\t")
hr_cers = hsb_rand["CER scores"]
avg_hr_cer = np.mean(hr_cers)
print(f"Average CER for hsb_rand is {avg_hr_cer:.2f}")

# lg
lg_fam = pd.read_csv("/scratch/jliebl/ASR_ComputEL/ASR_ComputEL/Results/lg_fam_eval_results_lg.tsv", sep="\t")
lf_cers = lg_fam["CER scores"]
avg_lf_cer = np.mean(lf_cers)
print(f"Average CER for lg_fam is {avg_lf_cer:.2f}")

lg_geo = pd.read_csv("/scratch/jliebl/ASR_ComputEL/ASR_ComputEL/Results/lg_geo_eval_results_lg.tsv", sep="\t")
lg_cers = lg_geo["CER scores"]
avg_lg_cer = np.mean(lg_cers)
print(f"Average CER for lg_geo is {avg_lg_cer:.2f}")

lg_phon = pd.read_csv("/scratch/jliebl/ASR_ComputEL/ASR_ComputEL/Results/lg_phon_eval_results_lg.tsv", sep="\t")
lp_cers = lg_phon["CER scores"]
avg_lp_cer = np.mean(lp_cers)
print(f"Average CER for lg_phon is {avg_lp_cer:.2f}")

lg_rand = pd.read_csv("/scratch/jliebl/ASR_ComputEL/ASR_ComputEL/Results/rand_eval_results_lg.tsv", sep="\t")
lr_cers = lg_rand["CER scores"]
avg_lr_cer = np.mean(lr_cers)
print(f"Average CER for lg_rand is {avg_lr_cer:.2f}")

# tt
tt_fam = pd.read_csv("/scratch/jliebl/ASR_ComputEL/ASR_ComputEL/Results/tt_fam_eval_results_tt.tsv", sep="\t")
tf_cers = tt_fam["CER scores"]
avg_tf_cer = np.mean(tf_cers)
print(f"Average CER for tt_fam is {avg_tf_cer}")

tt_geo = pd.read_csv("/scratch/jliebl/ASR_ComputEL/ASR_ComputEL/Results/tt_geo_eval_results_tt.tsv", sep="\t")
tg_cers = tt_geo["CER scores"]
avg_tg_cer = np.mean(tg_cers)
print(f"Average CER for tt_geo is {avg_tg_cer}")

tt_phon = pd.read_csv("/scratch/jliebl/ASR_ComputEL/ASR_ComputEL/Results/tt_phon_eval_results_tt.tsv", sep="\t")
tp_cers = tt_phon["CER scores"]
avg_tp_cer = np.mean(tp_cers)
print(f"Average CER for tt_phon is {avg_tp_cer:.2f}")

tt_rand = pd.read_csv("/scratch/jliebl/ASR_ComputEL/ASR_ComputEL/Results/rand_eval_results_tt.tsv", sep="\t")
tr_cers = tt_rand["CER scores"]
avg_tr_cer = np.mean(tr_cers)
print(f"Average CER for tt_rand is {avg_tr_cer:.2f}")