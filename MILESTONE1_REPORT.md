# Meilenstein 1 — Kurzreport: Domain & Baseline

_Stand: 2026-09-29_

## Problemverständnis
Ziel ist die zweistufige Klassifikation landwirtschaftlicher Kulturpflanzen anhand hyperspektraler Satellitendaten (EO-1 Hyperion): zunächst die Crop-Art (corn, soybean, winter_wheat, cotton, rice), anschließend das Growing Stage innerhalb dieser Crop-Klasse (Critical, Emerge_VEarly, Mature_Senesc, Early_Mid, Late, Harvest). Die Herausforderung liegt in der hohen Dimensionalität der Spektralbänder, teilweise unbrauchbaren Bändern und einer stark unausgeglichenen Klassenverteilung.

## Data Understanding (siehe `eda.ipynb`)
- `train.csv`: 5591 Zeilen, 198 Spektralband-Spalten (`X427`…`X2395`) plus `AEZ`, `Month`, `Crop`, `Stage`.
- **67 von 198 Spektralband-Spalten sind zu 100 % leer.** Die betroffenen Wellenlängenbereiche (~925–973nm, ~1104–1165nm, ~1326–1508nm, ~1770–2052nm, ~2355–2395nm) entsprechen bekannten Wasserdampf-Absorptionsbereichen bzw. verrauschten SWIR-Randbereichen des Hyperion-Sensors — fachlich plausibel unbrauchbar.
- Nur 6 Spalten haben vereinzelt fehlende Werte (<0,5 % je Spalte); jede Zeile hat mindestens einen NaN-Wert, zeilenweises Löschen ist daher keine Option.
- Klassenverteilung ist unausgeglichen: Crop (corn 2097, soybean 1669, winter_wheat 1073, cotton 659, rice 93), Stage (Critical 1541, Emerge_VEarly 1038, Mature_Senesc 989, Early_Mid 982, Late 861, Harvest 180).
- Die mittleren Spektralsignaturen unterscheiden sich sichtbar zwischen den Crops (siehe Plot in `eda.ipynb`) — die Klassifikationsaufgabe ist auf Basis der Bänder grundsätzlich lösbar.

## Preprocessing-Entscheidungen
- Die 67 komplett leeren Spektralband-Spalten werden **verworfen**, nicht imputiert (kein Informationsgehalt, würde reine Rauschwerte einbringen).
- Für die verbleibenden vereinzelten Lücken (<0,5 %) genügt **Median-Imputation** (`SimpleImputer`).
- Features: die verbleibenden 131 nutzbaren Spektralbänder plus `AEZ` und `Month` (133 Features insgesamt), skaliert mit `StandardScaler`.

## Baseline-Ansatz
Für die Baseline wurde die einfachste in der Aufgabenstellung genannte Variante gewählt: **zwei unabhängige `RandomForestClassifier`-Modelle** (eines für Crop, eines für Stage) auf denselben Features, mit `class_weight="balanced"` wegen des Klassenungleichgewichts. Stratifizierter 70/30 Train-Test-Split (stratifiziert auf `Crop`). Der Vergleich mit hierarchischen oder gemeinsamen Klassifikationsstrategien ist laut Aufgabenstellung expliziter Bestandteil von Meilenstein 2, nicht der Baseline.

## Ergebnisse

| Zielgröße | Balanced Accuracy | Macro F1 |
|---|---|---|
| Crop  | 0.778 | 0.810 |
| Stage | 0.769 | 0.788 |

Beide Baseline-Modelle liegen deutlich über dem Zufallsniveau (Crop: 5 Klassen → 20 % Zufall; Stage: 6 Klassen → 16,7 % Zufall) und bestätigen, dass die Spektralbänder eine tragfähige Grundlage für beide Klassifikationsaufgaben bieten. Details und Confusion Matrices in `baseline_model.ipynb`.

## Limitationen & nächste Schritte (Meilenstein 2)
- Kein Vergleich der Modellierungsstrategien (hierarchisch vs. gemeinsame Klasse vs. separate Modelle) — folgt in M2.
- Kein Hyperparameter-Tuning, keine Cross-Validation — nur ein einfacher Train/Test-Split für die Baseline.
- Klassenungleichgewicht nur über `class_weight` adressiert, keine weiteren Resampling-Strategien geprüft.
- Keine Feature-Selektion/Dimensionsreduktion (z. B. PCA) — relevant für Meilenstein 3 (Dateninput minimieren).
