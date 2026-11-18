
from pathlib import Path
import pandas as pd

# Cesta k novej CSV tabulke
CSV_PATH = Path(r"C:\Users\adam.demjanovic\Desktop\im_table.csv")

df = pd.read_csv(CSV_PATH)

print("========== ZAKLADNE INFORMACIE ==========")
print("Pocet riadkov:", len(df))
print("Pocet stlpcov:", len(df.columns))
print("Pocet unikatnych EKG:", df["ecg_id"].nunique())
print("Pocet unikatnych pacientov:", df["patient_id"].nunique())

print("\n========== CHYBAJUCE HODNOTY ==========")
print(df.isna().sum())

print("\n========== DUPLICITNE EKG ==========")
print("Pocet duplicitnych ecg_id:", df["ecg_id"].duplicated().sum())

print("\n========== POCET ZAZNAMOV PODLA STRAT_FOLD ==========")
print(df["strat_fold"].value_counts().sort_index())

print("\n========== POCET ZAZNAMOV PODLA TYPU INFARKTU ==========")

# Jeden zaznam moze obsahovat viac kodov.
codes = df["typ_infarktu"].dropna().str.split("; ").explode()
print(codes.value_counts())

print("\n========== KONTROLA CIEST K FEATURE MAPAM ==========")

path_columns = [
    "path_to_feature_map_lvl_4",
    "path_to_feature_map_lvl_8",
    "path_to_feature_map_lvl_16",
]

for column in path_columns:
    exists = df[column].apply(
        lambda path: isinstance(path, str) and Path(path).is_file()
    )

    print(f"{column}:")
    print("  Existujuce subory:", int(exists.sum()))
    print("  Chybajuce subory:", int((~exists).sum()))

print("\n========== HOTOVO ==========")
