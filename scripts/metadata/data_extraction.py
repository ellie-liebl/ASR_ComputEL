'''
J. Elizabeth Liebl; jliebl@gmu.edu
03/03/2026
ASR for ComputEL
Data Extraction
'''

import pandas as pd

import pandas as pd

id_df = pd.read_csv("ASR_ICLDC/rel_langs_w_ids.tsv", sep="\t", dtype=str)
gl_df = pd.read_csv("ASR_ICLDC/Glottolog_Data/values.csv", dtype=str)
gl_langs_df = pd.read_csv("ASR_ICLDC/Glottolog_Data/languages.csv", dtype=str)

# normalize glottocodes
id_df["Glottolog_ID"] = id_df["Glottolog_ID"].astype(str).str.strip().str.lower()

gl_langs_df["Glottocode"] = gl_langs_df["Glottocode"].astype(str).str.strip().str.lower()

# --- latitude/longitude: merge by Glottocode (correct key) ---
coords = gl_langs_df[["Glottocode", "Latitude", "Longitude"]].copy()

out = id_df.merge(coords, left_on="Glottolog_ID", right_on="Glottocode", how="left")
out = out.drop(columns=["Glottocode"])

# --- classification/family: values.csv uses IDs like "<glottocode>-classification"
gl_df["ID"] = gl_df["ID"].astype(str).str.strip()
gl_df["Value"] = gl_df["Value"].astype(str).str.strip()

out["classification_id"] = out["Glottolog_ID"] + "-classification"

# Map ID -> Value directly (no reverse lookup)
class_map = gl_df.set_index("ID")["Value"]
out["Classification"] = out["classification_id"].map(class_map)

out = out.drop(columns=["classification_id"])

# Optional: report which didn’t match
missing_class = out.loc[out["Classification"].isna(), "Glottolog_ID"].unique()
missing_coords = out.loc[out["Latitude"].isna() | out["Longitude"].isna(), "Glottolog_ID"].unique()

print("Missing Classification for:", missing_class[:25], "..." if len(missing_class) > 25 else "")
print("Missing coords for:", missing_coords[:25], "..." if len(missing_coords) > 25 else "")

out.to_csv("ASR_ICLDC/rel_langs_w_ids_and_data.tsv", sep="\t", index=False)

# id_df = pd.read_csv("ASR_ICLDC/rel_langs_w_ids.tsv", sep="\t")
# ph_df = pd.read_csv("ASR_ICLDC/Phoible_Data/cldf/values.csv")
# gl_df = pd.read_csv("ASR_ICLDC/Glottolog_Data/values.csv")
# gl_langs_df = pd.read_csv("ASR_ICLDC/Glottolog_Data/languages.csv")

# gl_fam_dict = dict(zip(gl_df["Value"], gl_df["ID"]))
# gl_long_dict = dict(zip(gl_langs_df["Longitude"], gl_langs_df["Glottocode"]))
# gl_lat_dict = dict(zip(gl_langs_df["Latitude"], gl_langs_df["Glottocode"]))

# def get_key_from_value(d, val):
#     for key, value in d.items():
#         if value == val:
#             return key
#     return None # Return None if value not found

# gl_ids = id_df["Glottolog_ID"]
# ph_ids = id_df["Phoible_ID"]

# families = []
# longs = []
# lats = []

# for i in gl_ids:
#     id_class = f"{i}-classification"
#     family = get_key_from_value(gl_fam_dict, id_class)
#     long = get_key_from_value(gl_long_dict, i)
#     lat = get_key_from_value(gl_lat_dict, i)
#     families.append(family)
#     longs.append(long)
#     lats.append(lat)

# id_df["Classification"] = families
# id_df["Latitude"] = lats
# id_df["Longitude"] = longs

# id_df.to_csv("ASR_ICLDC/rel_langs_w_ids_and_data.tsv", sep="\t", index=False)