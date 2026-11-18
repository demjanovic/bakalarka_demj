"""Baseline klasifikatory (logisticka regresia, naive Bayes) nad feature mapami
segmentacnej siete pre detekciu infarktu myokardu (PTB-XL).

Feature mapa jedneho EKG ma tvar (1, 48, L), L = 1250 / 625 / ~312 pre lvl 4 / 8 / 16.
Z kazdej mapy sa agregaciou cez poziciu (os L) spravi vektor priznakov na EKG:
48 kanalov x n_statistik.

Pouzitie (Windows, mapy na disku E:):
    python train_baseline.py                   # nastavenia z bloku CONFIG nizsie
    python train_baseline.py --level 8         # prepisanie z prikazoveho riadka
    python train_baseline.py --level 4 --cv    # krizova validacia cez foldy 1-9
    python train_baseline.py --final-test      # az na konci: vyhodnotenie na foldu 10

Priznaky sa raz vypocitaju a ulozia do cache (features_lvl{N}_{stats}.npz), dalsie
behy su rychle - modely a nastavenia mozes menit bez opatovneho nacitania 30 GB.
"""
import argparse
import hashlib
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, roc_auc_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler

# =============================================================================
# CONFIG - tu si menis nastavenia
# =============================================================================
CONFIG = dict(
    table="full_table.csv",          # tabulka z build_full_table.py
    level=16,                         # 4, 8 alebo 16  (ktora feature mapa)
    # Statistiky pocitane cez poziciu pre kazdy z 48 kanalov. Dostupne:
    #   mean, std, min, max, absmean, p05, p25, p50, p75, p95, energy, skew, kurt
    stats=["mean", "std", "min", "max", "absmean"],
    # Uprava priznakov pred modelom:
    #   "standard"  - StandardScaler (z-skore)
    #   "quantile"  - QuantileTransformer -> normalne rozdelenie (dobre pre naive Bayes)
    #   "power"     - PowerTransformer (Yeo-Johnson)
    #   "none"      - ziadna
    scaling="standard",
    pca_components=None,             # napr. 20 alebo 0.95 (podiel variancie); None = bez PCA
    train_folds=list(range(1, 9)),   # 1-8 trenovanie
    val_fold=9,                      # 9 validacia
    test_fold=10,                    # 10 test (pouzije sa len s --final-test)
    cv_folds=list(range(1, 10)),     # foldy pre --cv (test fold 10 ostava nedotknuty)
    # Ak su mapy na inom mieste nez v tabulke, nahrad prefix:
    path_prefix_from=None,           # napr. "E:\\"
    path_prefix_to=None,             # napr. "D:\\data\\"
    # Pri 'path_uncertain=1' (13 EKG) sa skusi aj susedny priecinok:
    try_alt_folders=True,
    cache_dir="cache",
    n_jobs_loading=1,                # citanie .pt je zvycajne I/O-bound; 1 je bezpecne
    seed=42,
)

# Modely na porovnanie. Pridavaj/menit hyperparametre podla chuti.
def make_models(seed):
    return {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logreg_C1": LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"),
        "logreg_C0.1": LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"),
        "logreg_C0.01": LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced"),
        "gaussian_nb": GaussianNB(),
    }


# =============================================================================
# Nacitanie feature mapy
# =============================================================================
def load_map(path):
    """Vrati np.ndarray (48, L) z .pt suboru."""
    import torch
    t = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(t, dict):                      # keby bol ulozeny ako slovnik
        t = next(iter(t.values()))
    a = t.detach().float().numpy()
    return a.reshape(a.shape[-2], a.shape[-1])   # (1,48,L) alebo (48,L) -> (48,L)


# =============================================================================
# Agregacia cez poziciu: (48, L) -> (48 * n_stats,)
# =============================================================================
def _stat(a, name):
    if name == "mean":    return a.mean(1)
    if name == "std":     return a.std(1)
    if name == "min":     return a.min(1)
    if name == "max":     return a.max(1)
    if name == "absmean": return np.abs(a).mean(1)
    if name == "energy":  return (a ** 2).mean(1)
    if name.startswith("p") and name[1:].isdigit():
        return np.percentile(a, int(name[1:]), axis=1)
    if name in ("skew", "kurt"):
        m = a.mean(1, keepdims=True); s = a.std(1, keepdims=True) + 1e-8
        z = (a - m) / s
        return (z ** 3).mean(1) if name == "skew" else (z ** 4).mean(1) - 3
    raise ValueError(f"Neznama statistika: {name}")


def aggregate(a, stats):
    return np.concatenate([_stat(a, s) for s in stats]).astype(np.float32)


def feature_names(stats, n_ch=48):
    return [f"ch{c:02d}_{s}" for s in stats for c in range(n_ch)]


# =============================================================================
# Tvorba tabulky priznakov (s cache)
# =============================================================================
def resolve_path(row, col, cfg):
    p = row[col]
    if cfg["path_prefix_from"]:
        p = p.replace(cfg["path_prefix_from"], cfg["path_prefix_to"] or "", 1)
    if os.path.exists(p) or not (cfg["try_alt_folders"] and row["path_uncertain"] == 1):
        return p
    base = os.path.basename(p)
    root = os.path.dirname(os.path.dirname(p))
    for f in ("Features_For_BT", "Features_For_BT1", "Features_For_BT2", "Features_For_BT3"):
        alt = os.path.join(root, f, base)
        if os.path.exists(alt):
            return alt
    return p


def build_features(df, cfg):
    col = f"path_to_feature_map_lvl_{cfg['level']}"
    key = hashlib.md5(("|".join(cfg["stats"]) + str(len(df))).encode()).hexdigest()[:8]
    os.makedirs(cfg["cache_dir"], exist_ok=True)
    cache = os.path.join(cfg["cache_dir"], f"features_lvl{cfg['level']}_{key}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        if np.array_equal(z["ecg_id"], df["ecg_id"].values):
            print(f"[cache] nacitane {cache}")
            return z["X"], z["ok"]

    X = np.full((len(df), 48 * len(cfg["stats"])), np.nan, dtype=np.float32)
    ok = np.zeros(len(df), dtype=bool)
    t0 = time.time()
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            X[i] = aggregate(load_map(resolve_path(row, col, cfg)), cfg["stats"])
            ok[i] = True
        except Exception as e:  # chybajuci/poskodeny subor nezhodi cely beh
            if (~ok).sum() <= 5:
                print(f"  [!] ecg_id={row['ecg_id']}: {type(e).__name__}: {e}")
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(df)}  ({time.time() - t0:.0f}s)")
    np.savez_compressed(cache, X=X, ok=ok, ecg_id=df["ecg_id"].values)
    print(f"[cache] ulozene {cache}  (nacitanych {ok.sum()}/{len(df)})")
    return X, ok


# =============================================================================
# Modely a vyhodnotenie
# =============================================================================
def make_pipeline(model, cfg):
    steps = []
    sc = cfg["scaling"]
    if sc == "standard":   steps.append(("scale", StandardScaler()))
    elif sc == "quantile": steps.append(("scale", QuantileTransformer(
        output_distribution="normal", n_quantiles=1000, random_state=cfg["seed"])))
    elif sc == "power":    steps.append(("scale", PowerTransformer()))
    elif sc != "none":     raise ValueError(f"scaling={sc}")
    if cfg["pca_components"]:
        steps.append(("pca", PCA(n_components=cfg["pca_components"], random_state=cfg["seed"])))
    steps.append(("clf", clone(model)))
    return Pipeline(steps)


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        roc_auc=roc_auc_score(y, p),
        pr_auc=average_precision_score(y, p),
        sens=tp / max(tp + fn, 1),
        spec=tn / max(tn + fp, 1),
        brier=brier_score_loss(y, p),
    )


def youden_threshold(y, p):
    """Prah maximalizujuci senzitivitu + specificitu (hladany na VALIDACII, nie na teste)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    return thr[np.argmax(tpr - fpr)]


def fmt(res):
    df = pd.DataFrame(res).T
    return df.round(3).to_string()


def run_holdout(df, X, cfg, eval_fold, label):
    tr = df.strat_fold.isin(cfg["train_folds"]).values
    ev = (df.strat_fold == eval_fold).values
    y = df.y.values
    out = {}
    for name, model in make_models(cfg["seed"]).items():
        pipe = make_pipeline(model, cfg).fit(X[tr], y[tr])
        p_tr = pipe.predict_proba(X[tr])[:, 1]
        p = pipe.predict_proba(X[ev])[:, 1]
        thr = 0.5 if name.startswith("dummy") else youden_threshold(y[tr], p_tr)
        m = metrics(y[ev], p, thr)
        m["train_auc"] = roc_auc_score(y[tr], p_tr)
        out[name] = m
    print(f"\n=== {label} (fold {eval_fold}, n={ev.sum()}, infarktov={y[ev].sum()}) ===")
    print("prah = Youden na trenovacich datach; train_auc vs roc_auc ukazuje pretrenovanie")
    print(fmt(out))
    return out


def run_cv(df, X, cfg):
    y = df.y.values
    rows = {n: [] for n in make_models(cfg["seed"])}
    for k in cfg["cv_folds"]:
        tr = df.strat_fold.isin([f for f in cfg["cv_folds"] if f != k]).values
        ev = (df.strat_fold == k).values
        for name, model in make_models(cfg["seed"]).items():
            pipe = make_pipeline(model, cfg).fit(X[tr], y[tr])
            p = pipe.predict_proba(X[ev])[:, 1]
            rows[name].append([roc_auc_score(y[ev], p), average_precision_score(y[ev], p)])
    print(f"\n=== Krizova validacia cez foldy {cfg['cv_folds']} (lvl {cfg['level']}) ===")
    res = {}
    for n, r in rows.items():
        r = np.array(r)
        res[n] = dict(roc_auc_mean=r[:, 0].mean(), roc_auc_std=r[:, 0].std(),
                      pr_auc_mean=r[:, 1].mean(), pr_auc_std=r[:, 1].std())
    print(fmt(res))
    return res


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table"); ap.add_argument("--level", type=int)
    ap.add_argument("--stats", help="napr. mean,std,min,max")
    ap.add_argument("--scaling"); ap.add_argument("--pca", type=float)
    ap.add_argument("--cv", action="store_true", help="krizova validacia cez foldy 1-9")
    ap.add_argument("--final-test", action="store_true",
                    help="vyhodnotenie na teste (fold 10) - pouzi az na konci")
    a = ap.parse_args()
    cfg = dict(CONFIG)
    if a.table: cfg["table"] = a.table
    if a.level: cfg["level"] = a.level
    if a.stats: cfg["stats"] = a.stats.split(",")
    if a.scaling: cfg["scaling"] = a.scaling
    if a.pca: cfg["pca_components"] = int(a.pca) if a.pca >= 1 else a.pca
    warnings.filterwarnings("ignore")

    df = pd.read_csv(cfg["table"])
    print(f"lvl={cfg['level']} stats={cfg['stats']} scaling={cfg['scaling']} pca={cfg['pca_components']}")
    X, ok = build_features(df, cfg)
    if not ok.all():
        print(f"[!] {(~ok).sum()} EKG sa nepodarilo nacitat - vynechane")
        df, X = df[ok].reset_index(drop=True), X[ok]
    assert np.isfinite(X).all(), "Priznaky obsahuju NaN/inf"

    run_holdout(df, X, cfg, cfg["val_fold"], f"VALIDACIA lvl {cfg['level']}")
    if a.cv:
        run_cv(df, X, cfg)
    if a.final_test:
        run_holdout(df, X, cfg, cfg["test_fold"], f"TEST lvl {cfg['level']}")


if __name__ == "__main__":
    main()