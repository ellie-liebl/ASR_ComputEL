'''
J. Elizabeth Liebl; jliebl@gmu.edu
03/11/2026
ASR for ComputEL
Phoible Citations
'''
import pandas as pd
import bibtexparser

p_id_df = pd.read_csv("ASR_ICLDC/rel_langs_full_df.tsv", sep="\t")
p_vals = pd.read_csv("ASR_ICLDC/Phoible_Data/cldf/values.csv")
all_bibs = "ASR_ICLDC/Phoible_Data/cldf/sources.bib"
out_bib = "ASR_ICLDC/inventory_sources.bib"

filtered_values = p_vals[p_vals["Language_ID"].isin(p_id_df["Phoible_ID"])]

bib_keys = set(filtered_values["Source"])
all_list_keys = []
for b in bib_keys:
    b_list = b.split(";")
    for l in b_list:
        all_list_keys.append(l)

all_keys = set(all_list_keys)
print(len(all_keys))

with open(all_bibs, "r", encoding="utf-8") as f:
    bib_db = bibtexparser.load(f)


selected_entries = [entry for entry in bib_db.entries if entry["ID"] in all_keys]


found_keys = {entry["ID"] for entry in selected_entries}
missing_keys = all_keys - found_keys

print(f"Requested: {len(all_keys)}")
print(f"Found: {len(found_keys)}")
print(f"Missing: {len(missing_keys)}")
if missing_keys:
    print("Missing keys:")
    for k in sorted(missing_keys):
        print(k)

new_db = bibtexparser.bibdatabase.BibDatabase()
new_db.entries = selected_entries

writer = bibtexparser.bwriter.BibTexWriter()
writer.indent = "    "
writer.order_entries_by = ("ID",)

with open(out_bib, "w", encoding="utf-8") as f:
    f.write(writer.write(new_db))

print(f"Wrote {len(selected_entries)} entries to {out_bib}")