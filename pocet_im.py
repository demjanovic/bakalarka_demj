
from pathlib import Path
import ast
import pandas as pd

# Uprav cesty k svojim suborom
DATABASE_PATH = Path(r"C:\Users\adam.demjanovic\Desktop\ptbxl_database.csv")
STATEMENTS_PATH = Path(r"C:\Users\adam.demjanovic\Desktop\scp_statements.csv")

database = pd.read_csv(DATABASE_PATH)
statements = pd.read_csv(STATEMENTS_PATH, index_col=0)

# Vyberieme diagnosticke kody a ich diagnosticke triedy.
diagnostic_statements = statements[
    statements["diagnostic"] == 1
]

code_to_class = diagnostic_statements[
    "diagnostic_class"
].to_dict()

def get_classes(value):
    codes = ast.literal_eval(value) if isinstance(value, str) else {}
    return {
        code_to_class[code]
        for code in codes
        if code in code_to_class
    }

database["classes"] = database["scp_codes"].apply(get_classes)

# Zaznam sa oznaci ako MI, ak ma aspon jeden diagnosticky
# kod namapovany do oficialnej triedy MI.
mi_mask = database["classes"].apply(
    lambda classes: "MI" in classes
)

print("Pocet vsetkych EKG:", len(database))
print("EKG s diagnostickou triedou MI:", int(mi_mask.sum()))
print("EKG bez diagnostickej triedy MI:", int((~mi_mask).sum()))
print("Pocet roznych pacientov s MI:",
      database.loc[mi_mask, "patient_id"].nunique())
print("Pocet EKG s MI aj dalsou diagnostickou triedou:",
      int(database.loc[mi_mask, "classes"].apply(
          lambda classes: len(classes) > 1
      ).sum()))
