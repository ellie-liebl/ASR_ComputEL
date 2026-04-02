'''
J. Elizabeth Liebl; jliebl@gmu.edu
03/11/2026
ASR for ComputEL
Strategy Analysis
'''

# Import Tools
import pandas as pd
import numpy as np

# Import Data
# Upper Sorbian
hsb_fam = pd.read_csv("ASR_ICLDC/Results/hsb_fam_eval_results_hsb.tsv", sep="\t")
hsb_geo = pd.read_csv("ASR_ICLDC/Results/hsb_geo_eval_results_hsb.tsv", sep="\t")
hsb_phon = pd.read_csv("ASR_ICLDC/Results/hsb_phon_eval_results_hsb.tsv", sep="\t")
hsb_rand = pd.read_csv("ASR_ICLDC/Results/rand_eval_results_hsb.tsv", sep="\t")

# Luganda 
lg_fam = pd.read_csv("ASR_ICLDC/Results/lg_fam_eval_results_lg.tsv", sep="\t")
lg_geo = pd.read_csv("ASR_ICLDC/Results/lg_geo_eval_results_lg.tsv", sep="\t")
lg_phon = pd.read_csv("ASR_ICLDC/Results/lg_phon_eval_results_lg.tsv", sep="\t")
lg_rand = pd.read_csv("ASR_ICLDC/Results/rand_eval_results_lg.tsv", sep="\t")

# Tatar
tt_fam = pd.read_csv("ASR_ICLDC/Results/tt_fam_eval_results_tt.tsv", sep="\t")
tt_geo = pd.read_csv("ASR_ICLDC/Results/tt_geo_eval_results_tt.tsv", sep="\t")
tt_phon = pd.read_csv("ASR_ICLDC/Results/tt_phon_eval_results_tt.tsv", sep="\t")
tt_rand = pd.read_csv("ASR_ICLDC/Results/rand_eval_results_tt.tsv", sep="\t")

# CER scores
hf_cers = list(hsb_fam["CER scores"])
hg_cers = list(hsb_geo["CER scores"])
hp_cers = list(hsb_phon["CER scores"])
hr_cers = list(hsb_rand["CER scores"])
lf_cers = list(lg_fam["CER scores"])
lg_cers = list(lg_geo["CER scores"])
lp_cers = list(lg_phon["CER scores"])
lr_cers = list(lg_rand["CER scores"])
tf_cers = list(tt_fam["CER scores"])
tg_cers = list(tt_geo["CER scores"])
tp_cers = list(tt_phon["CER scores"])
tr_cers = list(tt_rand["CER scores"])

# Sum and Mean by Strat
fam_cers = hf_cers + lf_cers + tf_cers
geo_cers = hg_cers + lg_cers + tg_cers
phon_cers = hp_cers + lp_cers + tp_cers
rand_cers = hr_cers + lr_cers + tr_cers

fam_mean = np.mean(fam_cers)
geo_mean = np.mean (geo_cers)
phon_mean = np.mean(phon_cers)
rand_mean = np.mean(rand_cers)

print(f"Family Mean CER: {fam_mean:.3f}")
print(f"Geographic Mean CER: {geo_mean:.3f}")
print(f"Phonetic Mean CER: {phon_mean:.3f}")
print(f"Random Mean CER: {rand_mean:.3f}")

# Improvement Over Random by lang+strat
hr_mean = np.mean(hr_cers)
lr_mean = np.mean(lr_cers)
tr_mean = np.mean(tr_cers)

hf_mean = np.mean(hf_cers)
lf_mean = np.mean(lf_cers)
tf_mean = np.mean(tf_cers)
hg_mean = np.mean(hg_cers)
lg_mean = np.mean(lg_cers)
tg_mean = np.mean(tg_cers)
hp_mean = np.mean(hp_cers)
lp_mean = np.mean(lp_cers)
tp_mean = np.mean(tp_cers)

def improv_over_rand(test, rand):
    improvement = (rand - test) / rand
    return improvement

hf_improv = improv_over_rand(hf_mean, hr_mean)
print(f"hsb Fam improvement over random: {hf_improv:.3f}")

hg_improv = improv_over_rand(hg_mean, hr_mean)
print(f"hsb Geo improvement over random: {hg_improv:.3f}")

hp_improv = improv_over_rand(hp_mean, hr_mean)
print(f"hsb Phon improvement over random: {hp_improv:.3f}")

lf_improv = improv_over_rand(lf_mean, lr_mean)
print(f"lg Fam improvement over random: {lf_improv:.3f}")

lg_improv = improv_over_rand(lg_mean, lr_mean)
print(f"lg Geo improvement over random: {lg_improv:.3f}")

lp_improv = improv_over_rand(lp_mean, lr_mean)
print(f"lg Phon improvement over random: {lp_improv:.3f}")

tf_improv = improv_over_rand(tf_mean, tr_mean)
print(f"tt Fam improvement over random: {tf_improv:.3f}")

tg_improv = improv_over_rand(tg_mean, tr_mean)
print(f"tt Geo improvement over random: {tg_improv:.3f}")

tp_improv = improv_over_rand(tp_mean, tr_mean)
print(f"tt Phon improvement over random: {tp_improv:.3f}")

