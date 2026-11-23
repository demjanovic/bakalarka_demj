"""Poskladanie prehladnej tabulky z ulozenych vysledkov (cache/cv_lvl*.csv a
cache/holdout_lvl*_fold10.csv), ktore vytvara first_train.py.

Pouzitie (v priecinku, kde je cache/):
    python make_table.py
Vytvori results_table.csv (vsetky metriky) a vypise prehladne tabulky
(ROC-AUC, F1, senzitivita, specificita) pre CV aj TEST.

Stlpce: CV = priemer +- std cez foldy 1-9 (k-nasobna krizova validacia),
        TEST = vyhodnotenie na foldu 10 (model trenovany na foldoch 1-8).
Prah pre F1 / senzitivitu / specificitu = Youden na trenovacich datach.
"""
import glob
import os
import re

import pandas as pd

CACHE = "cache"
METRICS = ["roc_auc", "pr_auc", "f1", "acc", "sens", "spec"]
SHOW = ["roc_auc", "f1", "sens", "spec"]           # metriky v prehladnom vypise
LABELS = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1", "acc": "Accuracy",
          "sens": "Senzitivita", "spec": "Specificita"}
NAMES = {"dummy_prior": "Dummy (prior)", "logreg_C0.1": "Logisticka regresia (C=0.1)",
         "logreg_C1": "Logisticka regresia (C=1)", "logreg_C0.01": "Logisticka regresia (C=0.01)",
         "gaussian_nb": "Naive Bayes", "gaussian_nb_pca40": "Naive Bayes + PCA 40",
         "hist_gb": "Gradient boosting"}
ORDER = list(NAMES)

rows = []
levels = sorted({int(re.search(r"lvl(\d+)", f).group(1))
                 for f in glob.glob(os.path.join(CACHE, "cv_lvl*.csv"))
                 + glob.glob(os.path.join(CACHE, "holdout_lvl*_fold10.csv"))})
if not levels:
    raise SystemExit("V cache/ nie su ziadne vysledky (cv_lvl*.csv, holdout_lvl*_fold10.csv).")

for lvl in levels:
    cv_f = os.path.join(CACHE, f"cv_lvl{lvl}.csv")
    te_f = os.path.join(CACHE, f"holdout_lvl{lvl}_fold10.csv")
    cv = pd.read_csv(cv_f) if os.path.exists(cv_f) else None
    te = pd.read_csv(te_f).set_index("model") if os.path.exists(te_f) else None
    models = set()
    if cv is not None: models |= set(cv.model)
    if te is not None: models |= set(te.index)
    for m in sorted(models, key=lambda x: ORDER.index(x) if x in ORDER else 99):
        r = {"Uroven": f"{lvl}x", "Model": NAMES.get(m, m)}
        for c in METRICS:
            if cv is not None and m in set(cv.model):
                g = cv[cv.model == m][c]
                r[f"CV {LABELS[c]}"] = f"{g.mean():.3f} +- {g.std(ddof=1):.3f}"
            else:
                r[f"CV {LABELS[c]}"] = ""
            r[f"TEST {LABELS[c]}"] = f"{te.loc[m, c]:.3f}" if te is not None and m in te.index else ""
        rows.append(r)

tab = pd.DataFrame(rows)
tab.to_csv("results_table.csv", index=False, encoding="utf-8-sig")  # utf-8-sig: Excel otvori spravne

# prehladne vypisy
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
print("\n=== K-nasobna krizova validacia (foldy 1-9): priemer +- std ===")
print(tab[["Uroven", "Model"] + [f"CV {LABELS[c]}" for c in SHOW]].to_string(index=False))
print("\n=== TEST (fold 10) ===")
print(tab[["Uroven", "Model"] + [f"TEST {LABELS[c]}" for c in SHOW]].to_string(index=False))
print("\n[ulozene] results_table.csv (obsahuje aj PR-AUC a accuracy)")
