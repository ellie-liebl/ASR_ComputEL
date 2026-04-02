'''
J. Elizabeth Liebl; jliebl@gmu.edu
03/11/2026
ASR for ComputEL
Phoible Stuff
'''

import pandas as pd
import re
from collections import defaultdict

def clean_segment_label(seg):
    if "|" in seg:
        seg = seg.split("|")[0]

    x = re.sub("t͡s", "ʦ", seg)
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
    return clean

def collapse_inventory_columns(phoible_df, lang_col="Language_ID"):
    groups = defaultdict(list)

    for col in phoible_df.columns:
        if col == lang_col:
            continue

        cleaned_col = clean_segment_label(col)
        groups[cleaned_col].append(col)

    collapsed_df = pd.DataFrame()
    collapsed_df[lang_col] = phoible_df[lang_col]

    for cleaned_col, original_cols in groups.items():
        collapsed_df[cleaned_col] = phoible_df[original_cols].max(axis=1)

    return collapsed_df, groups

phoible_df = pd.read_csv("ASR_ICLDC/rel_langs_phoible_inventories.tsv", sep="\t")

collapsed_df, groups = collapse_inventory_columns(phoible_df)
print(collapsed_df.head())
print(groups)

collapsed_df.to_csv("ASR_ICLDC/collapsed_rel_langs_phoible_inventories.tsv", sep="\t", index=False)