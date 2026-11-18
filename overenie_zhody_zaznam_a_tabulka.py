
# -*- coding: utf-8 -*-

"""
Kontrola prepojenia PTB-XL tabulky s feature mapami.
Samotne .pt subory sa nenacitavaju.
"""

from pathlib import Path
from collections import defaultdict
import ast
import re

import pandas as pd


# ============================================================
# 1. NASTAVENIE CIEST
# ============================================================

# SEM ZADAJ SKUTOCNU CESTU K CSV SUBORU
CSV_PATH = Path(
    r"C:\\Users\\adam.demjanovic\\Desktop\\ptbxl_database.csv"
)

# SEM ZADAJ SKUTOCNE CESTY K STYROM PRIECINKOM
FEATURE_DIRS = [
    Path(r"E:\Features_For_BT"),
    Path(r"E:\Features_For_BT1"),
    Path(r"E:\Features_For_BT2"),
    Path(r"E:\Features_For_BT3"),
]

EXPECTED_LEVELS = {4, 8, 16}

FILE_PATTERN = re.compile(
    r"^(\d+)_hr_decomposition_lvl_(4|8|16)\.pt$"
)


# ============================================================
# 2. NACITANIE TABULKY
# ============================================================

print("=" * 60)
print("1. ANOTACNA TABULKA PTB-XL")
print("=" * 60)

if not CSV_PATH.is_file():
    raise FileNotFoundError(
        "CSV subor sa nenasiel: " + str(CSV_PATH)
        + "\nSkontroluj nastavenie CSV_PATH."
    )

df = pd.read_csv(CSV_PATH, low_memory=False)

print("Pocet riadkov:", len(df))
print("Pocet stlpcov:", len(df.columns))
print("Nazvy stlpcov:", list(df.columns))

if "ecg_id" not in df.columns:
    raise ValueError("V tabulke chyba stlpec ecg_id.")

numeric_ids = pd.to_numeric(df["ecg_id"], errors="coerce")
table_ids = set(numeric_ids.dropna().astype(int))

print("Unikatne ECG ID:", df["ecg_id"].nunique())
print("Pocet ciselnych ECG ID:", len(table_ids))
print("Pocet duplicitnych riadkov ECG ID:",
      int(df["ecg_id"].duplicated().sum()))

columns_to_show = [
    col for col in
    ["ecg_id", "patient_id", "filename_hr", "scp_codes"]
    if col in df.columns
]

print("\nPrvych 10 riadkov tabulky:")
print(df[columns_to_show].head(10).to_string(index=False))


# ============================================================
# 3. PREHLAD ANOTACII
# ============================================================

print("\n" + "=" * 60)
print("2. ANOTACIE")
print("=" * 60)

if "scp_codes" in df.columns:

    def parse_scp_codes(value):
        if isinstance(value, dict):
            return value

        if not isinstance(value, str):
            return {}

        try:
            result = ast.literal_eval(value)
            if isinstance(result, dict):
                return result
        except (ValueError, SyntaxError):
            pass

        return {}

    parsed_codes = df["scp_codes"].apply(parse_scp_codes)

    all_codes = set()

    for codes in parsed_codes:
        all_codes.update(str(code) for code in codes.keys())

    print("Pocet roznych SCP kodov:", len(all_codes))
    print("Zoznam SCP kodov:")
    print(", ".join(sorted(all_codes)))

    # Orientacny zoznam castych kodov infarktu.
    # Neskor overime vsetky kody podla scp_statements.csv.
    mi_codes = {
        "AMI", "IMI", "ASMI", "ALMI", "LMI",
        "PMI", "IPMI"
    }

    mi_mask = parsed_codes.apply(
        lambda codes: any(
            str(code).upper() in mi_codes
            for code in codes
        )
    )

    print("\nPredbezny pocet zaznamov s vybranym kodom MI:",
          int(mi_mask.sum()))
    print("Ostatne zaznamy:", int((~mi_mask).sum()))
    print("POZOR: Pocet MI je zatial iba orientacny.")

else:
    print("Stlpec scp_codes neexistuje.")


# ============================================================
# 4. PREHLADANIE FEATURE MAP
# ============================================================

print("\n" + "=" * 60)
print("3. FEATURE MAPY")
print("=" * 60)

feature_files = defaultdict(list)
unrecognized_files = []
total_pt_files = 0

for folder in FEATURE_DIRS:

    print("\nPriecinok:", folder)

    if not folder.is_dir():
        print("UPOZORNENIE: Priecinok neexistuje.")
        continue

    folder_count = 0

    for file_path in folder.iterdir():

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() != ".pt":
            continue

        total_pt_files += 1
        folder_count += 1

        match = FILE_PATTERN.fullmatch(file_path.name)

        if match is None:
            unrecognized_files.append(str(file_path))
            continue

        ecg_id = int(match.group(1))
        level = int(match.group(2))

        feature_files[(ecg_id, level)].append(file_path)

    print("Pocet .pt suborov:", folder_count)

print("\nSpolu .pt suborov:", total_pt_files)
print("Subory s nerozpoznanym nazvom:",
      len(unrecognized_files))

if unrecognized_files:
    print("\nPrvych 10 nerozpoznanych suborov:")
    for name in unrecognized_files[:10]:
        print(name)


# ============================================================
# 5. KONTROLA UROVNI A DUPLICIT
# ============================================================

print("\n" + "=" * 60)
print("4. KONTROLA UROVNI DEKOMPOZICIE")
print("=" * 60)

levels_by_id = defaultdict(set)
duplicate_keys = {}

for key, paths in feature_files.items():

    ecg_id, level = key
    levels_by_id[ecg_id].add(level)

    if len(paths) > 1:
        duplicate_keys[key] = paths

feature_ids = set(levels_by_id.keys())

missing_levels = {}

for ecg_id, levels in levels_by_id.items():

    missing = EXPECTED_LEVELS - levels

    if missing:
        missing_levels[ecg_id] = sorted(missing)

complete_level_count = sum(
    1
    for levels in levels_by_id.values()
    if levels == EXPECTED_LEVELS
)

print("Unikatne ECG ID vo feature mapach:", len(feature_ids))
print("ECG ID so vsetkymi tromi urovnami:", complete_level_count)
print("ECG ID s chybajucou urovnou:", len(missing_levels))
print("Duplicitne kombinacie ECG ID a uroven:",
      len(duplicate_keys))

if missing_levels:
    print("\nPrvych 20 ID s chybajucimi urovnami:")

    for ecg_id, missing in list(missing_levels.items())[:20]:
        print("ECG ID:", ecg_id, "- chyba uroven:", missing)

if duplicate_keys:
    print("\nPrvych 10 duplicitnych kombinacii:")

    for key, paths in list(duplicate_keys.items())[:10]:
        print("ECG ID:", key[0], "- uroven:", key[1])

        for path in paths:
            print("  ", path)


# ============================================================
# 6. POROVNANIE S TABULKOU
# ============================================================

print("\n" + "=" * 60)
print("5. PREPOJENIE S TABULKOU PTB-XL")
print("=" * 60)

matched_ids = table_ids & feature_ids
table_without_features = table_ids - feature_ids
features_without_table = feature_ids - table_ids

print("ECG ID v tabulke:", len(table_ids))
print("ECG ID vo feature mapach:", len(feature_ids))
print("ECG ID v oboch zdrojoch:", len(matched_ids))
print("ECG ID v tabulke bez feature map:", len(table_without_features))
print("ECG ID vo feature mapach bez anotacie:", len(features_without_table))

if table_without_features:
    print("\nPrvych 20 ID z tabulky bez feature map:")
    print(sorted(table_without_features)[:20])

if features_without_table:
    print("\nPrvych 20 ID bez zaznamu v tabulke:")
    print(sorted(features_without_table)[:20])


# ============================================================
# 7. KOMPLETNE PREPOJENE ZAZNAMY
# ============================================================

print("\n" + "=" * 60)
print("6. KOMPLETNE PREPOJENE ZAZNAMY")
print("=" * 60)

complete_ids = set()

for ecg_id in matched_ids:

    if levels_by_id[ecg_id] != EXPECTED_LEVELS:
        continue

    has_duplicates = any(
        key[0] == ecg_id
        for key in duplicate_keys
    )

    if not has_duplicates:
        complete_ids.add(ecg_id)

print("Kompletne prepojene ECG ID:", len(complete_ids))
print("Prvych 20 ID:", sorted(complete_ids)[:20])


# ============================================================
# 8. ZHRNUTIE
# ============================================================

print("\n" + "=" * 60)
print("KONIEC KONTROLY")
print("=" * 60)

print("Kontrola dokoncena.")
print("Obsah .pt suborov sa nenacitaval.")
print("Ziadne subory neboli upravene ani vymazane.")
