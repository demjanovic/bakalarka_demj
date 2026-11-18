from pathlib import Path
import ast
import re
import pandas as pd


# --------------------------------------------------
# 1. CESTY K SUBOROM
# --------------------------------------------------

DATABASE_PATH = Path(r"C:\Users\adam.demjanovic\Desktop\ptbxl_database.csv")
STATEMENTS_PATH = Path(r"C:\Users\adam.demjanovic\Desktop\scp_statements.csv")

FEATURE_FOLDERS = [
    Path(r"E:\Features_For_BT"),
    Path(r"E:\Features_For_BT1"),
    Path(r"E:\Features_For_BT2"),
    Path(r"E:\Features_For_BT3"),
]

OUTPUT_PATH = Path(r"C:\Users\adam.demjanovic\Desktop\im_table.csv")


# --------------------------------------------------
# 2. VYBRANE KODY INFARKTU
# --------------------------------------------------

MI_CODES = {
    "IMI",
    "ASMI",
    "ILMI",
    "AMI",
    "ALMI",
    "LMI",
    "IPLMI",
    "IPMI",
    "PMI",
}


# --------------------------------------------------
# 3. NACITANIE ANOTACII
# --------------------------------------------------

database = pd.read_csv(DATABASE_PATH)
statements = pd.read_csv(STATEMENTS_PATH, index_col=0)

# Overime potrebne stlpce
required_columns = {
    "ecg_id",
    "patient_id",
    "scp_codes",
    "strat_fold",
}

missing_columns = required_columns - set(database.columns)

if missing_columns:
    raise ValueError(
        f"V databaze chybaju stlpce: {missing_columns}"
    )

if "description" not in statements.columns:
    raise ValueError(
        "V scp_statements.csv chyba stlpec description."
    )


# --------------------------------------------------
# 4. VYHLADANIE FEATURE MAP
# --------------------------------------------------

# Pre kazde EKG ulozime cestu k urovniam 4, 8 a 16.
feature_paths = {}

pattern = re.compile(
    r"^(\d+)_hr_decomposition_lvl_(4|8|16)\.pt$"
)

for folder in FEATURE_FOLDERS:
    if not folder.exists():
        raise FileNotFoundError(
            f"Priecinok neexistuje: {folder}"
        )

    for file_path in folder.rglob("*.pt"):
        match = pattern.match(file_path.name)

        if not match:
            continue

        ecg_id = int(match.group(1))
        level = int(match.group(2))

        if ecg_id not in feature_paths:
            feature_paths[ecg_id] = {}

        if level in feature_paths[ecg_id]:
            raise ValueError(
                f"Duplicitna feature mapa: "
                f"EKG {ecg_id}, uroven {level}"
            )

        feature_paths[ecg_id][level] = str(
            file_path.resolve()
        )


# --------------------------------------------------
# 5. VYTVORENIE MAPOVANIA KOD -> POPIS
# --------------------------------------------------

# Index tabulky statements obsahuje SCP kody.
code_to_description = (
    statements["description"].to_dict()
)


# --------------------------------------------------
# 6. SPRACOVANIE JEDNEHO EKG ZAZNAMU
# --------------------------------------------------

def parse_scp_codes(value):
    """Prevedie text scp_codes na slovnik kodov."""

    if isinstance(value, dict):
        return value

    if not isinstance(value, str):
        return {}

    try:
        result = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return {}

    return result if isinstance(result, dict) else {}


rows = []

for _, record in database.iterrows():
    ecg_id = int(record["ecg_id"])

    scp_codes = parse_scp_codes(record["scp_codes"])

    # Vyberieme len nase pozadovane kody.
    matching_codes = [
        code
        for code in scp_codes
        if code in MI_CODES
    ]

    # Ak EKG nema ani jeden vybrany kod, preskocime ho.
    if not matching_codes:
        continue

    # Ziskame cesty k feature mapam.
    paths = feature_paths.get(ecg_id, {})

    # Nechceme do vystupu zaradit zaznam bez vsetkych
    # troch ocakavanych feature map.
    if not all(level in paths for level in (4, 8, 16)):
        print(
            f"UPOZORNENIE: EKG {ecg_id} nema "
            "vsetky tri feature mapy."
        )
        continue

    # Kod a jeho popis zostanu v rovnakom poradi.
    descriptions = [
        str(code_to_description.get(code, "Popis nenajdeny"))
        for code in matching_codes
    ]

    rows.append({
        "ecg_id": ecg_id,
        "patient_id": record["patient_id"],
        "typ_infarktu": "; ".join(matching_codes),
        "scp_description": "; ".join(descriptions),
        "strat_fold": record["strat_fold"],
        "path_to_feature_map_lvl_4": paths[4],
        "path_to_feature_map_lvl_8": paths[8],
        "path_to_feature_map_lvl_16": paths[16],
    })


# --------------------------------------------------
# 7. VYTVORENIE A ULOZENIE NOVEJ TABULKY
# --------------------------------------------------

result = pd.DataFrame(rows)

if result.empty:
    raise ValueError(
        "Nenasli sa ziadne zaznamy. Skontroluj cesty "
        "a nazvy SCP kodov."
    )

result = result.sort_values("ecg_id")

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

result.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)


# --------------------------------------------------
# 8. KONTROLA VYSLEDKOV
# --------------------------------------------------

print("\n========== VYSLEDKY ==========")

print("Pocet riadkov:", len(result))
print("Pocet unikatnych EKG:", result["ecg_id"].nunique())
print("Pocet unikatnych pacientov:", result["patient_id"].nunique())

print("\nPocet vyskytov jednotlivych kodov:")

code_counts = {
    code: sum(
        code in codes.split("; ")
        for codes in result["typ_infarktu"]
    )
    for code in sorted(MI_CODES)
}

for code, count in code_counts.items():
    print(f"{code}: {count}")

print("\nUlozene do:")
print(OUTPUT_PATH)

print("\nPrvych 5 riadkov:")
print(result.head().to_string(index=False))
