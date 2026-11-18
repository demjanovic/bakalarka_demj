import pandas as pd
import ast


# ============================================================
# 1. NACITANIE DATASETU
# ============================================================

file_path = "C:\\Users\\adam.demjanovic\\Desktop\\ptbxl_database.csv"

df = pd.read_csv(file_path)


# ============================================================
# 2. ZAKLADNE INFORMACIE
# ============================================================

print("=" * 70)
print("ZAKLADNE INFORMACIE O DATASETE")
print("=" * 70)

print(f"Pocet zaznamov: {len(df)}")
print(f"Pocet stlpcov:  {len(df.columns)}")

print("\nNazvy stlpcov:")
for i, column in enumerate(df.columns, start=1):
    print(f"{i:2}. {column}")


# ============================================================
# 3. DATOVE TYPY
# ============================================================

print("\n" + "=" * 70)
print("DATOVE TYPY")
print("=" * 70)

print(df.dtypes)


# ============================================================
# 4. CHYBAJUCE HODNOTY
# ============================================================

print("\n" + "=" * 70)
print("CHYBAJUCE HODNOTY")
print("=" * 70)

missing = df.isna().sum()

missing_table = pd.DataFrame({
    "missing_count": missing,
    "missing_percent": (missing / len(df) * 100).round(2)
})

print(missing_table)


# ============================================================
# 5. UNIKATNE HODNOTY
# ============================================================

print("\n" + "=" * 70)
print("POCET UNIKATNYCH HODNOT")
print("=" * 70)

for column in df.columns:
    print(f"{column:25} {df[column].nunique()}")


# ============================================================
# 6. PACIENTI
# ============================================================

print("\n" + "=" * 70)
print("PACIENTI")
print("=" * 70)

unique_patients = df["patient_id"].nunique()

print(f"Pocet EKG zaznamov:       {len(df)}")
print(f"Pocet unikatnych pacientov: {unique_patients}")
print(f"Priemerny pocet EKG/pacient: "
      f"{len(df) / unique_patients:.2f}")

ecg_per_patient = df.groupby("patient_id").size()

print("\nPocet EKG podla pacienta:")
print(ecg_per_patient.describe())


# ============================================================
# 7. VEK
# ============================================================

print("\n" + "=" * 70)
print("VEK")
print("=" * 70)

print(df["age"].describe())


# ============================================================
# 8. POHLAVIE
# ============================================================

print("\n" + "=" * 70)
print("POHLAVIE")
print("=" * 70)

print(df["sex"].value_counts(dropna=False))


# ============================================================
# 9. STRATIFIKOVANE FOLDY
# ============================================================

print("\n" + "=" * 70)
print("STRAT_FOLD")
print("=" * 70)

print(df["strat_fold"].value_counts().sort_index())


# ============================================================
# 10. SCP DIAGNOSTICKE KODY
# ============================================================

print("\n" + "=" * 70)
print("SCP DIAGNOSTICKE KODY")
print("=" * 70)

all_scp_codes = {}

for value in df["scp_codes"].dropna():

    try:
        codes = ast.literal_eval(value)

        for code, probability in codes.items():

            if code not in all_scp_codes:
                all_scp_codes[code] = 0

            all_scp_codes[code] += 1

    except (ValueError, SyntaxError):
        pass


scp_table = pd.DataFrame(
    list(all_scp_codes.items()),
    columns=["code", "count"]
)

scp_table = scp_table.sort_values(
    "count",
    ascending=False
)

print(scp_table.to_string(index=False))


# ============================================================
# 11. MI KODY
# ============================================================

print("\n" + "=" * 70)
print("MI DIAGNOSTICKE KODY")
print("=" * 70)

# Kody pouzivane v PTB-XL pre myocardial infarction
mi_codes = {
    "AMI",
    "ALMI",
    "ASMI",
    "IMI",
    "ILMI",
    "LMI",
    "IPMI",
    "IPLMI"
}

mi_count = 0
mi_code_counts = {code: 0 for code in mi_codes}

for value in df["scp_codes"].dropna():

    try:
        codes = ast.literal_eval(value)

        found_mi = False

        for code in codes:

            if code in mi_codes:

                found_mi = True
                mi_code_counts[code] += 1

        if found_mi:
            mi_count += 1

    except (ValueError, SyntaxError):
        pass


print("Pocty jednotlivych MI kodov:")

for code, count in sorted(
    mi_code_counts.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{code:8} {count}")


print(f"\nPocet EKG s aspon jednym MI kodom: {mi_count}")
print(f"Pocet EKG bez MI kodu:              {len(df) - mi_count}")


# ============================================================
# 12. INFARCTION STADIUM
# ============================================================

print("\n" + "=" * 70)
print("INFARCTION STADIUM")
print("=" * 70)

print("\ninfarction_stadium1:")
print(df["infarction_stadium1"].value_counts(dropna=False))

print("\ninfarction_stadium2:")
print(df["infarction_stadium2"].value_counts(dropna=False))


# ============================================================
# 13. DUPLICITY
# ============================================================

print("\n" + "=" * 70)
print("DUPLICITY")
print("=" * 70)

print(f"Duplicitne ecg_id: "
      f"{df['ecg_id'].duplicated().sum()}")

print(f"Duplicitne patient_id: "
      f"{df['patient_id'].duplicated().sum()}")


# ============================================================
# 14. KVALITA DAT
# ============================================================

print("\n" + "=" * 70)
print("KVALITA DAT")
print("=" * 70)

print("\nPocet zaznamov s chybajucimi hodnotami:")

rows_with_missing = df.isna().any(axis=1).sum()

print(f"{rows_with_missing} / {len(df)} "
      f"({rows_with_missing / len(df) * 100:.2f} %)")

print("\nPocet kompletne vyplnenych zaznamov:")

complete_rows = df.notna().all(axis=1).sum()

print(f"{complete_rows} / {len(df)} "
      f"({complete_rows / len(df) * 100:.2f} %)")


# ============================================================
# 15. VYSLEDNY PREHLAD
# ============================================================

print("\n" + "=" * 70)
print("SUMAR")
print("=" * 70)

print(f"Pocet EKG:              {len(df)}")
print(f"Pocet pacientov:         {unique_patients}")
print(f"MI EKG:                  {mi_count}")
print(f"Non-MI EKG:              {len(df) - mi_count}")
print(f"MI podiel:               {mi_count / len(df) * 100:.2f} %")
print(f"Pocet foldov:             {df['strat_fold'].nunique()}")
print(f"Duplicitne ecg_id:        {df['ecg_id'].duplicated().sum()}")
print(f"Priemerny pocet EKG/pacient: "
      f"{len(df) / unique_patients:.2f}")

print("=" * 70)
print("HOTOVO")
print("=" * 70)