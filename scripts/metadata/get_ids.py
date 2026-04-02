'''
J. Elizabeth Liebl; jliebl@gmu.edu
03/02/2026
ASR for ComputEL
ID Extraction
'''

import pandas as pd

ph_langs_df = pd.read_csv("ASR_ICLDC/Phoible_Data/cldf/languages.csv")
gl_langs_df = pd.read_csv("ASR_ICLDC/Glottolog_Data/languages.csv")
rel_langs_df = pd.read_csv("ASR_ICLDC/rel_langs.tsv", sep="\t")

isos = rel_langs_df["ISO_639-3"]
names = rel_langs_df["Language_Name"]

col = "ISO639P3code" 

gl_ids = []
ph_ids = []

gl_dict_iso = dict(zip(gl_langs_df["ID"], gl_langs_df["ISO639P3code"]))
gl_dict_name = dict(zip(gl_langs_df["ID"], gl_langs_df["Name"]))
ph_dict_iso = dict(zip(ph_langs_df["ID"], ph_langs_df["ISO639P3code"]))
ph_dict_name = dict(zip(ph_langs_df["ID"], ph_langs_df["Name"]))

def get_key_from_value(d, val):
    for key, value in d.items():
        if value == val:
            return key
    return None # Return None if value not found

for i, n in zip(isos, names):
    gl_id = get_key_from_value(gl_dict_iso, i)
    if gl_id == None:
        gl_id = get_key_from_value(gl_dict_name, n)
    gl_ids.append(gl_id)    
    ph_id = get_key_from_value(ph_dict_iso, i)
    if ph_id == None:
        ph_id = get_key_from_value(ph_dict_name, n)
    ph_ids.append(ph_id)  

rel_langs_df["Glottolog_ID"] = gl_ids
rel_langs_df["Phoible_ID"] = ph_ids

rel_langs_df.to_csv("ASR_ICLDC/rel_langs_w_ids.tsv", sep="\t", index=False)