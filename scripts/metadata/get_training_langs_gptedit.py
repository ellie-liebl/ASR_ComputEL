'''
J. Elizabeth Liebl; jliebl@gmu.edu
03/03/2026
ASR for ComputEL
Pick Training Langs
'''
# Import Tools
import pandas as pd
from geopy import distance
import itertools

# Import Data
data_df = pd.read_csv("ASR_ICLDC/rel_langs_full_df.tsv", sep="\t")
phoible_df = pd.read_csv("ASR_ICLDC/collapsed_rel_langs_phoible_inventories.tsv", sep="\t")

# Classification col to list
data_df['Class_Set'] = data_df['Classification'].apply(lambda x: x.split('/'))

# Set Targets
target_langs = ["Upper Sorbian", "Luganda", "Tatar"]
target_isos = ["hsb", "lug", "tat"]
target_ph_ids = ["uppe1395", "gand1255", "tata1255"]

# hsb
hsb_lat = data_df.loc[data_df["ISO_639.3"] == 'hsb', 'Latitude'].iloc[0]
hsb_long = data_df.loc[data_df["ISO_639.3"] == 'hsb', 'Longitude'].iloc[0]
hsb_class = data_df.loc[data_df["ISO_639.3"] == 'hsb', 'Class_Set'].iloc[0]

# lug
lug_lat = data_df.loc[data_df["ISO_639.3"] == 'lug', 'Latitude'].iloc[0]
lug_long = data_df.loc[data_df["ISO_639.3"] == 'lug', 'Longitude'].iloc[0]
lug_class = data_df.loc[data_df["ISO_639.3"] == 'lug', 'Class_Set'].iloc[0]

# tat
tat_lat = data_df.loc[data_df["ISO_639.3"] == 'tat', 'Latitude'].iloc[0]
tat_long = data_df.loc[data_df["ISO_639.3"] == 'tat', 'Longitude'].iloc[0]
tat_class = data_df.loc[data_df["ISO_639.3"] == 'tat', 'Class_Set'].iloc[0]

# Drop targets from data
n_df = data_df[data_df["ISO_639.3"] != "hsb"]
ne_df = n_df[n_df["ISO_639.3"] != "lug"]
new_df = ne_df[ne_df["ISO_639.3"] != "tat"].copy()

# Figure out the distances first
hsb_loc = (hsb_lat, hsb_long)
hsb_dists = {}

for i, lat, long in zip(new_df["ISO_639.3"], new_df["Latitude"], new_df["Longitude"]):
    loc = (lat, long)
    dist = distance.distance(hsb_loc, loc).km
    hsb_dists[i] = dist

sorted_hsb_dists = sorted(hsb_dists.items(), key=lambda item: item[1])
print("hsb")
print(sorted_hsb_dists[:5])

lug_loc = (lug_lat, lug_long)
lug_dists = {}

for i, lat, long in zip(new_df["ISO_639.3"], new_df["Latitude"], new_df["Longitude"]):
    loc = (lat, long)
    dist = distance.distance(lug_loc, loc).km
    lug_dists[i] = dist

sorted_lug_dists = sorted(lug_dists.items(), key=lambda item: item[1])
print("lug")
print(sorted_lug_dists[:5])

tat_loc = (tat_lat, tat_long)
tat_dists = {}

for i, lat, long in zip(new_df["ISO_639.3"], new_df["Latitude"], new_df["Longitude"]):
    loc = (lat, long)
    dist = distance.distance(tat_loc, loc).km
    tat_dists[i] = dist

sorted_tat_dists = sorted(tat_dists.items(), key=lambda item: item[1])
print("tat")
print(sorted_tat_dists[:5])

# Now figure out familial closeness
new_df["Class_Set"] = new_df["Class_Set"].apply(set)

def family_overlap_df(df, target_set, target_iso):
    out = df.copy()
    out["overlap_n"] = out["Class_Set"].apply(lambda s: len(s & target_set))
    out["target_size"] = len(target_set)
    out["candidate_size"] = out["Class_Set"].apply(len)
    out["union_n"] = out["Class_Set"].apply(lambda s: len(s | target_set))

    # Percent of target family path covered by candidate
    out["overlap_pct_target"] = out["overlap_n"] / out["target_size"] * 100

    # Optional: percent of candidate family path shared with target
    out["overlap_pct_candidate"] = out["overlap_n"] / out["candidate_size"] * 100

    # Optional: Jaccard as percent
    out["jaccard_pct"] = out["overlap_n"] / out["union_n"] * 100

    top = out.sort_values(
        ["overlap_n", "overlap_pct_target", "jaccard_pct"],
        ascending=False
    ).head(5)[
        ["ISO_639.3", "overlap_n", "overlap_pct_target",
         "overlap_pct_candidate", "jaccard_pct", "Class_Set"]
    ]

    print(target_iso)
    print(top.to_string(index=False))
    return top

hsb_set = set(hsb_class)
lug_set = set(lug_class)
tat_set = set(tat_class)

hsb_top5 = family_overlap_df(new_df, hsb_set, "hsb")
lug_top5 = family_overlap_df(new_df, lug_set, "lug")
tat_top5 = family_overlap_df(new_df, tat_set, "tat")

# Finally.... the inventories
# First column is Language_ID; the rest are segment columns
lang_col = "Language_ID"
segment_cols = [c for c in phoible_df.columns if c != lang_col]

# Build inventories: lang -> set(segments)
inventories = {}
for _, row in phoible_df.iterrows():
    lang = row[lang_col]
    segs = {seg for seg in segment_cols if row[seg] == 1}
    inventories[lang] = segs

per_target_top5 = {}

for tid in target_ph_ids:
    tset = inventories[tid]
    rows = []

    for lang, inv in inventories.items():
        if lang == tid:
            continue

        overlap_n = len(inv & tset)
        union_n = len(inv | tset)

        target_size = len(tset)
        candidate_size = len(inv)

        overlap_pct_target = (overlap_n / target_size * 100) if target_size else 0.0
        overlap_pct_candidate = (overlap_n / candidate_size * 100) if candidate_size else 0.0
        jaccard_pct = (overlap_n / union_n * 100) if union_n else 0.0

        rows.append({
            "Language_ID": lang,
            "overlap_n": overlap_n,
            "overlap_pct_target": overlap_pct_target,
            "overlap_pct_candidate": overlap_pct_candidate,
            "jaccard_pct": jaccard_pct
        })

    df = pd.DataFrame(rows).sort_values(
        ["overlap_n", "overlap_pct_target", "jaccard_pct"],
        ascending=False
    ).head(5)

    per_target_top5[tid] = df

for tid, df in per_target_top5.items():
    print(f"\nTop 5 for {tid}:")
    print(df.to_string(index=False))


'''
Distances:
hsb
[('ces', 159.81977236150396), ('pol', 300.03451565388656), ('slk', 434.1982973435943), ('slv', 555.1206453805974), ('hun', 615.4211949065221)]
lug
[('luo', 314.8141614479224), ('kin', 372.9029118587339), ('swa', 1158.7803803211807), ('bas', 2441.634534000651), ('hau', 2832.1328635045375)]
tat
[('chv', 127.67968288025996), ('rus', 360.75814232208756), ('bak', 538.3858351371216), ('est', 1434.3432661784982), ('ukr', 1448.6547089239534)]

Families
hsb
ISO_639.3  overlap_n  overlap_pct_target  overlap_pct_candidate  jaccard_pct                          
                                    Class_Set
      ces          5           83.333333              83.333333    71.428571           {czec1260, indo1319, west2792, clas1257, slav1255, balt1263}
      slk          5           83.333333              83.333333    71.428571           {czec1260, indo1319, west2792, clas1257, slav1255, balt1263}
      pol          5           83.333333              71.428571    62.500000 {indo1319, west2792, clas1257, slav1255, poli1262, lech1241, balt1263}
      rus          4           66.666667              80.000000    57.142857                     {indo1319, clas1257, east1426, slav1255, balt1263}
      slv          4           66.666667              66.666667    50.000000           {indo1319, clas1257, slav1255, west2804, sout3147, balt1263}
lug
ISO_639.3  overlap_n  overlap_pct_target  overlap_pct_candidate  jaccard_pct                          
                                                                                                Class_Set
      kin          9           81.818182              75.000000    64.285714           {east2731, atla1278, volt1241, benu1247, west2842, sout3152, rwan1241, bant1294, nort3203, grea1289, narr1281, kivu1239}
      swa          8           72.727273              61.538462    50.000000 {east2731, atla1278, volt1241, benu1247, momb1256, sout3152, bant1294, nort3203, nort3209, coas1317, swah1254, narr1281, saba1282}
      bas          6           54.545455              54.545455    37.500000                     {basa1292, basa1289, atla1278, bant1295, volt1241, basa1283, benu1247, sout3152, bant1294, narr1281, basa1290}
      yor          3           27.272727              27.272727    15.789474                     {yoru1244, east2738, sout3186, atla1278, nucl1747, volt1241, benu1247, defo1239, edea1234, lucu1239, edek1238}
      abk          0            0.000000               0.000000     0.000000                          
                                                                                     {abkh1243, abkh1242}
tat
ISO_639.3  overlap_n  overlap_pct_target  overlap_pct_candidate  jaccard_pct                          
                                    Class_Set
      bak          7          100.000000             100.000000   100.000000 {comm1245, bash1267, kipc1239, turk1311, kipc1240, nort2696, nort3424}
      kir          4           57.142857              66.666667    44.444444           {sout3396, east2791, comm1245, kipc1239, turk1311, kipc1240}
      uzb          3           42.857143              60.000000    33.333333                     {comm1245, uygh1241, uygh1240, turk1311, kipc1240}
      uig          3           42.857143              50.000000    30.000000           {comm1245, uygh1241, uygh1240, turk1311, kipc1240, uigh1243}
      sah          2           28.571429              66.666667    25.000000                          
               {comm1245, turk1311, nort2688}

Phonetic inventory overlap:
Top 5 for uppe1395:
Language_ID  overlap_n  overlap_pct_target  overlap_pct_candidate  jaccard_pct
   lith1251         35           85.365854              47.945205    44.303797
   roma1327         33           80.487805              55.000000    48.529412
   bulg1262         33           80.487805              46.478873    41.772152
   ukra1253         31           75.609756              67.391304    55.357143
   russ1263         28           68.292683              50.909091    41.176471

Top 5 for gand1255:
Language_ID  overlap_n  overlap_pct_target  overlap_pct_candidate  jaccard_pct
   luok1236         25           89.285714              44.642857    42.372881
   dutc1256         25           89.285714              29.761905    28.735632
   indo1316         24           85.714286              68.571429    61.538462
   basa1284         24           85.714286              58.536585    53.333333
   stan1289         24           85.714286              42.105263    39.344262

Top 5 for tata1255:
Language_ID  overlap_n  overlap_pct_target  overlap_pct_candidate  jaccard_pct
   bash1264         29           69.047619              54.716981    43.939394
   hind1269         27           64.285714              30.337079    25.961538
   uigh1240         26           61.904762              66.666667    47.272727
   poli1260         26           61.904762              57.777778    42.622951
   nort2641         26           61.904762              41.935484    33.333333

Random
   swe, ukr, abk, ron
   swed1254, ukra1253, abkh1244, roma1327
'''

# -----------------------------
# RANDOM MODEL RELATIONSHIPS
# -----------------------------

# One shared random model for all targets
random_isos = ["swe", "ukr", "abk", "ron"]
random_ph_ids = ["swed1254", "ukra1253", "abkh1244", "roma1327"]

# Target metadata
target_meta = {
    "hsb": {
        "loc": (hsb_lat, hsb_long),
        "class_set": set(hsb_class),
        "ph_id": "uppe1395"
    },
    "lug": {
        "loc": (lug_lat, lug_long),
        "class_set": set(lug_class),
        "ph_id": "gand1255"
    },
    "tat": {
        "loc": (tat_lat, tat_long),
        "class_set": set(tat_class),
        "ph_id": "tata1255"
    }
}

# ISO -> PHOIBLE ID mapping for the random model languages
random_iso_to_ph = dict(zip(random_isos, random_ph_ids))

def compute_random_model_relationships(target_iso, random_isos, iso_to_ph, inventories, candidate_df):
    target_loc = target_meta[target_iso]["loc"]
    target_class_set = target_meta[target_iso]["class_set"]
    target_ph_id = target_meta[target_iso]["ph_id"]
    target_inventory = inventories[target_ph_id]

    # Restrict metadata dataframe to the random-model languages
    sub_df = candidate_df[candidate_df["ISO_639.3"].isin(random_isos)].copy()

    rows = []

    for _, row in sub_df.iterrows():
        cand_iso = row["ISO_639.3"]
        cand_class_set = row["Class_Set"]   # already converted to set
        cand_loc = (row["Latitude"], row["Longitude"])

        # Geographic distance
        geo_km = distance.distance(target_loc, cand_loc).km

        # Family overlap
        fam_overlap_n = len(cand_class_set & target_class_set)
        fam_union_n = len(cand_class_set | target_class_set)
        fam_target_size = len(target_class_set)
        fam_candidate_size = len(cand_class_set)

        fam_overlap_pct_target = (fam_overlap_n / fam_target_size * 100) if fam_target_size else 0.0
        fam_overlap_pct_candidate = (fam_overlap_n / fam_candidate_size * 100) if fam_candidate_size else 0.0
        fam_jaccard_pct = (fam_overlap_n / fam_union_n * 100) if fam_union_n else 0.0

        # Phonological overlap
        cand_ph_id = iso_to_ph.get(cand_iso, None)

        if cand_ph_id in inventories:
            cand_inventory = inventories[cand_ph_id]

            ph_overlap_n = len(cand_inventory & target_inventory)
            ph_union_n = len(cand_inventory | target_inventory)
            ph_target_size = len(target_inventory)
            ph_candidate_size = len(cand_inventory)

            ph_overlap_pct_target = (ph_overlap_n / ph_target_size * 100) if ph_target_size else 0.0
            ph_overlap_pct_candidate = (ph_overlap_n / ph_candidate_size * 100) if ph_candidate_size else 0.0
            ph_jaccard_pct = (ph_overlap_n / ph_union_n * 100) if ph_union_n else 0.0
        else:
            ph_overlap_n = None
            ph_overlap_pct_target = None
            ph_overlap_pct_candidate = None
            ph_jaccard_pct = None

        rows.append({
            "target_iso": target_iso,
            "candidate_iso": cand_iso,
            "geo_km": geo_km,
            "family_overlap_n": fam_overlap_n,
            "family_overlap_pct_target": fam_overlap_pct_target,
            "family_overlap_pct_candidate": fam_overlap_pct_candidate,
            "family_jaccard_pct": fam_jaccard_pct,
            "phono_overlap_n": ph_overlap_n,
            "phono_overlap_pct_target": ph_overlap_pct_target,
            "phono_overlap_pct_candidate": ph_overlap_pct_candidate,
            "phono_jaccard_pct": ph_jaccard_pct
        })

    return pd.DataFrame(rows)

# Run for each target against the same random model
random_relationship_dfs = {}

for target_iso in ["hsb", "lug", "tat"]:
    out_df = compute_random_model_relationships(
        target_iso=target_iso,
        random_isos=random_isos,
        iso_to_ph=random_iso_to_ph,
        inventories=inventories,
        candidate_df=new_df
    )
    random_relationship_dfs[target_iso] = out_df

# Print detailed results
for target_iso, df in random_relationship_dfs.items():
    print(f"\nRandom model relationships for {target_iso}:")
    print(df.to_string(index=False))

'''
Random model relationships for hsb:
target_iso candidate_iso      geo_km  family_overlap_n  family_overlap_pct_target  family_jaccard_pct  phono_overlap_n  phono_overlap_pct_target  phono_jaccard_pct
       hsb           abk 2207.279530                 0                   0.000000            0.000000               23                 56.097561          26.436782
       hsb           ron  899.001506                 2                  33.333333           14.285714               33                 80.487805          48.529412
       hsb           swe  971.195045                 2                  33.333333           16.666667               17                 41.463415          27.868852
       hsb           ukr 1111.292442                 4                  66.666667           50.000000               31                 75.609756          55.357143

Random model relationships for lug:
target_iso candidate_iso      geo_km  family_overlap_n  family_overlap_pct_target  family_jaccard_pct  phono_overlap_n  phono_overlap_pct_target  phono_jaccard_pct
       lug           abk 4780.150811                 0                        0.0                 0.0               17                 60.714286          21.250000
       lug           ron 5124.401379                 0                        0.0                 0.0               22                 78.571429          33.333333
       lug           swe 6681.223815                 0                        0.0                 0.0               18                 64.285714          38.297872
       lug           ukr 5448.259462                 0                        0.0                 0.0               20                 71.428571          37.037037

Random model relationships for tat:
target_iso candidate_iso      geo_km  family_overlap_n  family_overlap_pct_target        family_jaccard_pct  phono_overlap_n  phono_overlap_pct_target           phono_jaccard_pct
       tat           abk 1527.543152                 0                        0.0                 0.0               23                 54.761905           26.136364
       tat           ron 2016.903428                 0                        0.0                 0.0               25                 59.523810           32.467532
       tat           swe 1918.519440                 0                        0.0                 0.0               19                 45.238095           31.666667
       tat           ukr 1448.654709                 0                        0.0                 0.0               23                 54.761905           35.384615
'''