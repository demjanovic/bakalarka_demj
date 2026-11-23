# Analýza informační hodnoty EKG příznaků ze segmentační neuronové sítě pro detekci infarktu myokardu

Bakalárska práca. Cieľom je zistiť, **koľko informácie o infarkte myokardu nesú feature mapy**
z internej vrstvy segmentačnej neurónovej siete aplikovanej na EKG (databáza PTB-XL), a ako sa to
mení s mierou kompresie máp (4x, 8x, 16x). Úloha je binárna klasifikácia EKG:
`y = 1` infarkt, `y = 0` všetko ostatné (zdravé aj iné patológie).

> Tento súbor slúži aj ako kontext pre AI agenta pracujúceho v príkazovom riadku.
> Sekcia **Pravidlá pre prácu** je záväzná.

---

## Dáta

- **PTB-XL**: 21 799 EKG, z toho **5 288 infarktov** (24,3 %). Zdroj: `ptbxl_database.csv`.
- **Feature mapy**: súbory `.pt` (PyTorch tenzor tvaru `(1, 48, L)`, float32), `L` = 1250 / 625 / ~312 pre
  úroveň kompresie 4x / 8x / 16x. 48 kanálov × L pozícií. Spolu cca 30 GB, **nie sú v repozitári**.
  Ležia na disku `E:\` v priečinkoch `Features_For_BT`, `Features_For_BT1`, `Features_For_BT2`,
  `Features_For_BT3` (rozdelené podľa rozsahu `ecg_id`), súbory
  `{ecg_id:05d}_hr_decomposition_lvl_{4|8|16}.pt`.
- **`full_table.csv`** (vytvára `build_full_table.py`): jeden riadok na EKG.

| Stĺpec | Význam |
|---|---|
| `ecg_id`, `patient_id` | identifikátory |
| `strat_fold` | fold 1 až 10 (z PTB-XL, delené podľa pacienta) |
| `y` | 1 = infarkt, 0 = nie |
| `typ_infarktu`, `scp_description` | typ infarktu (len pre `y=1`, inak prázdne) |
| `path_to_feature_map_lvl_4/8/16` | cesty k mapám |
| `path_uncertain` | 1 = priečinok pre toto EKG nie je istý (13 EKG, všetky `y=0`) |

- **Label**: `y=1`, ak `scp_codes` obsahuje aspoň jeden z kódov IMI, ASMI, ILMI, AMI, ALMI, LMI, IPLMI, IPMI, PMI.
  Zhoduje sa presne s tabuľkou infarktových EKG `im_table.csv`. Zahrnuté sú aj EKG s nízkou pravdepodobnosťou
  v `scp_codes` a bez overenia lekárom (okolo 1 900 z 5 288) – definícia infarktu sa s vedúcim ešte konzultuje.
- **Delenie**: foldy 1–8 trénovanie, 9 validácia, 10 test. Podiel infarktov je vo všetkých približne rovnaký.

## Súbory

| Súbor | Účel |
|---|---|
| `build_full_table.py` | zostaví `full_table.csv` z `ptbxl_database.csv` a `im_table.csv` |
| `first_train.py` | baseline: príznaky = štatistiky z máp, modely logistická regresia / naive Bayes (/ boosting), validácia, k-násobná CV, test |
| `train_cnn.py` | 1D CNN (PyTorch) nad celou mapou (48 × L) |
| `make_table.py` | poskladá výsledky z `cache/` do `results_table.csv` |
| `cache/` | medzivýsledky (príznaky `features_lvl*.npz`, mapy `maps_lvl*.npy`, výsledky `cv_*.csv`, `holdout_*.csv`) – **negitovať** |
| `cnn_runs/` | uložené CNN modely, história trénovania a predikcie – **negitovať** |

> `train_cnn.py` importuje baseline skript ako modul (riadok `import ... as tb`: používa `load_map`,
> `resolve_path`, `metrics`, `youden_threshold`, `fmt`, `CONFIG`). Názov importovaného modulu musí
> zodpovedať skutočnému názvu baseline súboru (`first_train`).

## Ako to spustiť

Všetko sa spúšťa **z priečinka projektu** (kde je `full_table.csv` a `cache/`), Windows, Python.
Knižnice: `numpy`, `pandas`, `scikit-learn`, `torch` (CUDA; vývojový stroj má RTX 4050).
Spúšťať interpreterom z `../.venv` (systémový Python nemá `torch` – mapy sa potom nenačítajú).

```
python build_full_table.py ptbxl_database.csv im_table.csv full_table.csv

# baseline: validácia (fold 9) + k-násobná CV (foldy 1-9) + test (fold 10), pre každú úroveň
python first_train.py --level 4  --segments 1 --cv --final-test --models dummy_prior,logreg_C0.1,gaussian_nb,gaussian_nb_pca40
python first_train.py --level 8  --segments 1 --cv --final-test --models dummy_prior,logreg_C0.1,gaussian_nb,gaussian_nb_pca40
python first_train.py --level 16 --segments 1 --cv --final-test --models dummy_prior,logreg_C0.1,gaussian_nb,gaussian_nb_pca40
python make_table.py                      # -> results_table.csv

# CNN
python train_cnn.py --level 8 --final-test
python train_cnn.py --level 8 --seed 1 --final-test   # opakovanie s iným seedom
```

Užitočné prepínače baseline: `--level`, `--segments`, `--stats mean,std,...`, `--scaling standard|quantile|power|none`,
`--pca`, `--models`, `--cv`, `--final-test`. Ďalšie nastavenia sú v bloku `CONFIG` na začiatku súboru
(napr. `path_prefix_from` / `path_prefix_to`, ak sa mapy presunú z `E:\`).

### Trvanie a cache

- **Prvé načítanie máp** jednej úrovne trvá ~10 minút (I/O z disku), potom sa použije cache.
  Cache sa hľadá podľa úrovne, zoznamu `stats` a počtu úsekov – ak sa zmenia, začne sa čítať odznova.
  Ak sa napriek tomu vo výpise objaví priebeh `1000/21799`, nastavenia sa líšia od minulého behu.
- **CNN**: prvý beh načíta mapy do `cache/maps_lvl{N}.npy` (float16; lvl 16 ≈ 0,65 GB, lvl 8 ≈ 1,3 GB,
  lvl 4 ≈ 2,6 GB), jedna epocha trvá desiatky sekúnd až minúty, celý beh zvyčajne okolo hodiny.

## Metodika vyhodnotenia

- Trénovanie na foldoch 1–8, ladenie na folde 9, **test (fold 10) sa používa iba raz na konci** (`--final-test`).
- K-násobná CV (`--cv`) beží cez foldy 1–9; fold 10 ostáva nedotknutý. Výsledok: priemer ± smerodajná odchýlka.
- Metriky: **ROC-AUC, PR-AUC, F1 (pre triedu infarkt), accuracy, senzitivita, špecificita**, Brier skóre.
  Hlavné porovnanie modelov sa robí podľa AUC a F1.
- **Accuracy je zavádzajúca**: dummy model („nikdy infarkt“) má ~0,76 pri 24 % infarktov.
- Prah pre F1/accuracy/senzitivitu/špecificitu: pri baseline Youden na trénovacích dátach, pri CNN
  Youden na validácii (preto sú tieto štyri čísla na validácii mierne optimistické). Brier pri CNN nie je
  kalibrovaný (vážená strata `pos_weight`) a nemá sa porovnávať s baseline.
- Príznaky baseline: pre každý z 48 kanálov štatistiky cez pozície (`mean, std, min, max, absmean`, voliteľne
  percentily, `energy`, `skew`, `kurt`), škálovanie `quantile`. Rozdelenie mapy na časové úseky
  (`--segments > 1`) sa neosvedčilo (nezarovnané pozície medzi EKG, pretrénovanie).

## Doterajšie výsledky (orientačné, jeden beh, môžu byť neaktuálne – pozri `results_table.csv`)

Test (fold 10), ROC-AUC:

| Model | 4x | 8x | 16x |
|---|---|---|---|
| Logistická regresia (štatistiky) | 0,766 | 0,771 | 0,751 |
| CNN | nespustené | 0,834 | 0,788 |

CNN na úrovni 8: validácia AUC 0,868 / F1 0,64, test AUC 0,834 / F1 0,61. Logistická regresia na úrovni 8: test F1 0,52.
Naive Bayes (~0,63) a boosting (~0,77) boli horšie než logistická regresia.

## Pravidlá pre prácu

1. **Fold 10 (test) nepoužívaj na ladenie** – ani na výber úrovne, modelu, prahu alebo hyperparametrov. Voľby sa robia podľa validácie / CV.
2. **Nemaž ani neprepisuj `cache/` a `cnn_runs/`** bez opýtania sa. Obnova cache stojí ~10 minút na úroveň. Radšej premenuj priečinok.
3. **Nič z toho nepatrí do gitu**: `cache/`, `cnn_runs/`, `*.pt`, `*.npz`, `*.npy`, `ptbxl_database.csv`. Over `.gitignore` a `git status` pred každým commitom. Dáta a feature mapy sa nenahrávajú.
4. **Pri porovnaní úrovní/modelov drž nastavenia rovnaké** (`stats`, `scaling`, počet úsekov, seed). Rozdiely medzi behmi < ~0,01 AUC sú v rámci šumu – na tvrdenia o rozdieloch používaj CV alebo viac seedov.
5. **Pred dlhými behmi** (načítanie máp ~10 min, CNN ~1 h, CNN CV) upozorni a pýtaj sa, nie spúšťaj zbytočne.
6. Nezavádzaj nové závislosti bez opýtania. Skripty majú zostať spustiteľné z príkazového riadka s prepínačmi.
7. Komentáre v kóde a výstupy skriptov sú po slovensky (bez diakritiky v kóde), čísla vo výstupoch zaokrúhľuj na 3 desatinné miesta.
8. Ak sa zmení význam metrík, delenie dát alebo definícia `y`, **zapíš to sem** do README.

## Známe obmedzenia a otvorené otázky

- Nie je známe, **na akých dátach sa trénovala segmentačná sieť** (ak na PTB-XL, môže to ovplyvniť férovosť testu).
- Definícia infarktu (všetky infarktové kódy vs. len overené / s vysokou pravdepodobnosťou) čaká na rozhodnutie vedúceho.
- 13 EKG má neistú cestu k mape (`path_uncertain=1`; ID 5480–5488, 10947–10948, 16421–16422). Skript skúša všetky priečinky.
- Referencia, voči ktorej sa „informačná hodnota“ meria (surové EKG? pôvodný výstup siete?), nie je určená.
- Význam 48 kanálov máp nie je známy (zvody? abstraktné príznaky?).
- Výsledky CNN sú zatiaľ z jedného seedu; CNN pre úroveň 4 a CNN k-násobná CV nie sú urobené.

## Ďalšie kroky

1. Doplniť baseline tabuľku pre 4x/8x/16x (CV + test) a CNN pre všetky úrovne s 3 seedmi.
2. **Analýza informačnej hodnoty** (jadro témy): ablácia / permutačná dôležitosť jednotlivých kanálov a častí mapy
   pre natrénovanú CNN (uložené modely v `cnn_runs/`).
3. Výsledky podľa typu infarktu (`typ_infarktu`) z uložených predikcií.
4. Prípadne k-násobná CV aj pre CNN, ak to vedúci vyžaduje.