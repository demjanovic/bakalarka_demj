"""1D CNN priamo nad feature mapami (48 kanalov x L pozicii) pre detekciu infarktu (PTB-XL).

Na rozdiel od train_baseline.py sa tu NEPOCITAJU rucne statistiky - siet dostane celu mapu
a sama sa nauci, ktore vzory (a na ktorej pozicii) su dolezite.

Pouzitie (v priecinku s full_table.csv, train_baseline.py a cache/):
    python train_cnn.py --level 16              # prvy beh: nacita mapy (dlho), potom trenuje
    python train_cnn.py --level 16              # dalsie behy: mapy z cache, rychlo
    python train_cnn.py --level 8
    python train_cnn.py --level 16 --final-test # vyhodnotenie na teste (fold 10) - az na konci

Mapy sa pri prvom behu ulozia do cache/maps_lvl{N}.npy (float16). Velkost cache:
lvl 16 ~ 0.65 GB, lvl 8 ~ 1.3 GB, lvl 4 ~ 2.6 GB. Rovnaku velkost zaberie aj RAM pri trenovani.

Skript pouziva funkcie z train_baseline.py (nacitanie .pt, cesty, metriky) - musi lezat vedla.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import first_train as tb  # load_map, resolve_path, metrics, youden_threshold, fmt, CONFIG

# =============================================================================
# CONFIG - tu si menis nastavenia
# =============================================================================
CFG = dict(
    table="full_table.csv",
    level=16,                        # 4, 8 alebo 16
    cache_dir="cache",
    out_dir="cnn_runs",              # sem sa ulozi model, historia trenovania a predikcie
    train_folds=list(range(1, 9)),
    val_fold=9,
    test_fold=10,                    # pouzije sa len s --final-test
    epochs=40,
    batch_size=128,
    lr=1e-3,
    weight_decay=1e-2,
    width=64,                        # zakladny pocet filtrov (dalsie bloky 2x, 2x, 4x)
    dropout=0.3,
    noise_std=0.05,                  # gaussovsky sum pridavany pocas trenovania (v jednotkach std)
    patience=8,                      # early stopping: kolko epoch bez zlepsenia val AUC
    amp=True,                        # zmiesana presnost na GPU (rychlejsie, menej pamate)
    seed=42,
)


# =============================================================================
# Mapy -> jedno velke pole v cache (float16)
# =============================================================================
def maps_files(cfg):
    base = os.path.join(cfg["cache_dir"], f"maps_lvl{cfg['level']}")
    return base + ".npy", base + "_ok.npy", base + "_ids.npy"


def build_maps(df, cfg):
    """Vrati (X: memmap/array (N, 48, L) float16, ok: bool maska nacitanych EKG)."""
    path, okpath, idpath = maps_files(cfg)
    if all(os.path.exists(p) for p in (path, okpath, idpath)):
        ids = np.load(idpath)
        if np.array_equal(ids, df["ecg_id"].values):
            print(f"[cache] nacitane {path}")
            return np.load(path, mmap_mode="r"), np.load(okpath)

    os.makedirs(cfg["cache_dir"], exist_ok=True)
    col = f"path_to_feature_map_lvl_{cfg['level']}"
    tcfg = dict(tb.CONFIG)
    arr, L, n_ch = None, None, None
    ok = np.zeros(len(df), dtype=bool)
    n_err, t0 = 0, time.time()
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            a = tb.load_map(tb.resolve_path(row, col, tcfg))
            if arr is None:
                n_ch, L = a.shape
                arr = np.lib.format.open_memmap(
                    path, mode="w+", dtype=np.float16, shape=(len(df), n_ch, L))
                print(f"tvar jednej mapy: ({n_ch}, {L})")
            if a.shape[0] != n_ch:
                raise ValueError(f"zly pocet kanalov {a.shape[0]}")
            if a.shape[1] >= L:                      # zjednotenie dlzky (orez / doplnenie nulami)
                a = a[:, :L]
            else:
                a = np.pad(a, ((0, 0), (0, L - a.shape[1])))
            arr[i] = a.astype(np.float16)
            ok[i] = True
        except Exception as e:
            n_err += 1
            if n_err <= 5:
                print(f"  [!] ecg_id={row['ecg_id']}: {type(e).__name__}: {e}")
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(df)}  ({time.time() - t0:.0f}s)")
    if arr is None:
        raise RuntimeError("Nepodarilo sa nacitat ziadnu mapu - skontroluj cesty (path_prefix_* v train_baseline.CONFIG).")
    arr.flush()
    np.save(okpath, ok)
    np.save(idpath, df["ecg_id"].values)
    print(f"[cache] ulozene {path}  (nacitanych {ok.sum()}/{len(df)})")
    return np.load(path, mmap_mode="r"), ok


def channel_stats(X, idx, chunk=256):
    """Priemer a std pre kazdy kanal cez trenovacie EKG a vsetky pozicie."""
    idx = np.sort(idx)
    C = X.shape[1]
    s, ss, n = np.zeros(C), np.zeros(C), 0
    for st in range(0, len(idx), chunk):
        b = np.asarray(X[idx[st:st + chunk]], dtype=np.float64)   # (b, C, L)
        s += b.sum((0, 2))
        ss += (b ** 2).sum((0, 2))
        n += b.shape[0] * b.shape[2]
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 0)) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


# =============================================================================
# Model: 4 konvolucne bloky + globalny pooling (priemer a maximum) + maly klasifikator
# =============================================================================
class Block(nn.Module):
    def __init__(self, cin, cout, k=7):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, k, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm1d(cout)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2, ceil_mode=True)

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))


class Net(nn.Module):
    def __init__(self, c_in, width, dropout):
        super().__init__()
        self.features = nn.Sequential(
            Block(c_in, width), Block(width, width * 2),
            Block(width * 2, width * 2), Block(width * 2, width * 4))
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(width * 4 * 2, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):                       # x: (batch, C, L)
        h = self.features(x)
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=1)   # globalny pooling cez poziciu
        return self.head(h).squeeze(1)          # logit


# =============================================================================
# Trenovanie / predikcia
# =============================================================================
def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):          # starsie verzie torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def predict(model, X, idx, mean_t, std_t, dev, bs, amp):
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        xb = X[idx[i:i + bs]].to(dev).float()
        xb = (xb - mean_t) / std_t
        with torch.autocast(device_type=dev.type, enabled=amp):
            out.append(torch.sigmoid(model(xb).float()).cpu())
    return torch.cat(out).numpy()


def auc_pair(y, p):
    from sklearn.metrics import average_precision_score, roc_auc_score
    return roc_auc_score(y, p), average_precision_score(y, p)


def run(cfg, final_test):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg["amp"]) and dev.type == "cuda"
    print(f"zariadenie: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))
    if dev.type == "cpu":
        print("[!] GPU nenajdene - trenovanie na CPU bude pomale (skus level 16).")
    torch.backends.cudnn.benchmark = True

    df = pd.read_csv(cfg["table"])
    X, ok = build_maps(df, cfg)
    if not ok.all():
        print(f"[!] {(~ok).sum()} EKG sa nepodarilo nacitat - vynechane")
    df = df[ok].reset_index(drop=True)
    X = np.asarray(X[ok])                                  # do RAM (float16)
    assert np.isfinite(X).all(), "Mapy obsahuju NaN/inf"
    y = df["y"].values.astype(np.float32)
    ecg = df["ecg_id"].values

    tr = np.where(df.strat_fold.isin(cfg["train_folds"]))[0]
    va = np.where(df.strat_fold == cfg["val_fold"])[0]
    te = np.where(df.strat_fold == cfg["test_fold"])[0]
    print(f"X: {X.shape} ({X.nbytes / 1e9:.2f} GB)  train={len(tr)} val={len(va)} test={len(te)}")

    mean, std = channel_stats(X, tr)
    mean_t = torch.from_numpy(mean).view(1, -1, 1).to(dev)
    std_t = torch.from_numpy(std).view(1, -1, 1).to(dev)
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    tr_t = torch.from_numpy(tr)

    model = Net(X.shape[1], cfg["width"], cfg["dropout"]).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"parametrov: {n_par:,}")
    pos_w = float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = make_scaler(amp)

    os.makedirs(cfg["out_dir"], exist_ok=True)
    tag = f"cnn_lvl{cfg['level']}_seed{cfg['seed']}"
    best_auc, best_state, best_ep, hist = -1.0, None, -1, []
    bs = cfg["batch_size"]
    for ep in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        perm = tr_t[torch.randperm(len(tr_t))]
        tot, nb = 0.0, 0
        for i in range(0, len(perm), bs):
            ib = perm[i:i + bs]
            if len(ib) < 2:                    # BatchNorm potrebuje aspon 2 vzorky
                continue
            xb = (Xt[ib].to(dev).float() - mean_t) / std_t
            if cfg["noise_std"] > 0:
                xb = xb + cfg["noise_std"] * torch.randn_like(xb)
            yb = yt[ib].to(dev)
            with torch.autocast(device_type=dev.type, enabled=amp):
                logit = model(xb)
            loss = lossf(logit.float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            nb += 1
        sched.step()

        p_va = predict(model, Xt, va, mean_t, std_t, dev, bs, amp)
        auc, pr = auc_pair(y[va], p_va)
        hist.append(dict(epoch=ep, train_loss=tot / max(nb, 1), val_auc=auc, val_pr_auc=pr))
        mark = ""
        if auc > best_auc:
            best_auc, best_ep, mark = auc, ep, "  *"
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epocha {ep:3d}  loss {tot / max(nb, 1):.4f}  val AUC {auc:.4f}  "
              f"PR-AUC {pr:.4f}  ({time.time() - t0:.0f}s){mark}")
        if ep - best_ep >= cfg["patience"]:
            print(f"early stopping (najlepsia epocha {best_ep})")
            break

    pd.DataFrame(hist).to_csv(os.path.join(cfg["out_dir"], f"{tag}_history.csv"), index=False)
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(cfg["out_dir"], f"{tag}.pt"))

    # --- vyhodnotenie najlepsieho modelu (podla val AUC) ---
    p_tr = predict(model, Xt, tr, mean_t, std_t, dev, bs, amp)
    p_va = predict(model, Xt, va, mean_t, std_t, dev, bs, amp)
    thr = tb.youden_threshold(y[va], p_va)       # prah z validacie
    m = tb.metrics(y[va], p_va, thr)
    m["train_auc"] = auc_pair(y[tr], p_tr)[0]
    print(f"\n=== VALIDACIA CNN lvl {cfg['level']} (fold {cfg['val_fold']}, n={len(va)}, "
          f"infarktov={int(y[va].sum())}, najlepsia epocha {best_ep}) ===")
    print(f"prah = {thr:.3f} (Youden na validacii, preto je f1/acc/sens/spec na validacii mierne optimisticke)")
    print(tb.fmt({"cnn": m}))
    np.savez_compressed(os.path.join(cfg["out_dir"], f"{tag}_val_pred.npz"),
                        ecg_id=ecg[va], y=y[va], p=p_va)

    if final_test:
        p_te = predict(model, Xt, te, mean_t, std_t, dev, bs, amp)
        mt = tb.metrics(y[te], p_te, thr)        # prah z VALIDACIE, test sa nepouziva na ladenie
        print(f"\n=== TEST CNN lvl {cfg['level']} (fold {cfg['test_fold']}, n={len(te)}, "
              f"infarktov={int(y[te].sum())}) ===")
        print(f"prah = {thr:.3f} (z validacie)")
        print(tb.fmt({"cnn": mt}))
        np.savez_compressed(os.path.join(cfg["out_dir"], f"{tag}_test_pred.npz"),
                            ecg_id=ecg[te], y=y[te], p=p_te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table"); ap.add_argument("--level", type=int)
    ap.add_argument("--epochs", type=int); ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float); ap.add_argument("--width", type=int)
    ap.add_argument("--dropout", type=float); ap.add_argument("--seed", type=int)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--final-test", action="store_true",
                    help="vyhodnotenie na teste (fold 10) - pouzi az na konci")
    a = ap.parse_args()
    cfg = dict(CFG)
    for k in ("table", "level", "epochs", "lr", "width", "dropout", "seed"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    if a.batch_size: cfg["batch_size"] = a.batch_size
    if a.no_amp: cfg["amp"] = False
    print(f"CNN: lvl={cfg['level']} epochs={cfg['epochs']} batch={cfg['batch_size']} "
          f"lr={cfg['lr']} width={cfg['width']} dropout={cfg['dropout']}")
    run(cfg, a.final_test)


if __name__ == "__main__":
    main()