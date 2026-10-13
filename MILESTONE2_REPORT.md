# Meilenstein 2 — Modellreife & Validierung

_Stand: 2026-10-13_

## Ausgangspunkt: Feedback zu Meilenstein 1
- Stratifizierung soll auch `Stage` berücksichtigen, nicht nur `Crop`.
- Zyklisches Encoding von `Month` prüfen.
- Weitere Modelle testen (Gradient Boosting Trees u. a.).
- `SimpleImputer` nicht mit Median, sondern mit den benachbarten Werten.
- Andere Split-Verhältnisse als 70/30 ausprobieren.
- Cross-Validation / saubere Validierung statt einfachem Train/Test-Split.

Details und Code: `milestone2_model.ipynb` (Laufzeit komplett ca. 14 Min. auf 8 Kernen).

## Preprocessing-Entscheidungen (neu ggü. M1)

**Spektral-Imputation:** Die 67 komplett leeren Spektralband-Spalten werden weiterhin verworfen. Für die verbleibenden 6 Spalten mit vereinzelten Lücken (max. 24 von 5591 Zeilen) ersetzen wir die globale Median-Imputation durch **zeilenweise lineare Interpolation entlang der Wellenlänge**: Da die Spaltennamen (`X<nm>`) die Wellenlänge kodieren und Reflektanzspektren glatt verlaufen, lässt sich eine fehlende Bande besser aus den benachbarten Bändern *derselben Messung* schätzen als aus dem Median über alle Messungen. Die Interpolation ist zeilenlokal → kein Leakage zwischen Train/Test. Ein `SimpleImputer(median)` bleibt als Absicherung in der Pipeline für etwaige Randfälle.

**Feature Engineering** (übernommen und überarbeitet aus `features.ipynb`, plus eigene Red-Edge-/Seneszenz-Indizes): 22 zeilenlokale Spektral-Features + AEZ One-Hot.
- Spektral-Statistiken (mean, std, max, min, Amplitude, Wellenlänge von Max/Min)
- Vegetationsindizes NDVI, SR, NDMI, NBR, ND NIR2/SWIR1, EVI (auf Prozent-Reflektanz angepasst)
- Gradienten (max. positive/negative Steigung, totale Variation), **pro nm** berechnet, damit die Lücken der entfernten Wasserbänder nicht dominieren
- neu: NDRE, CI red-edge, Red-Edge-Position (Guyot & Baret), PSRI (Seneszenz), PRI, NDWI nach Gao (Wassergehalt)
- nicht übernommen: 46 Differenzen/Ratios an Prozent-Positionen des Spektrums (ohne physikalische Bedeutung, teils Duplikate) und die Breiten der Hoch-/Tiefbereiche (über die Bandlücken hinweg irreführend)

**Month-Encoding:** `Month` reicht nur von 5 (Mai) bis 10 (Oktober) — kein Jahreswechsel, den eine zyklische Kodierung auflösen müsste. sin/cos-Encoding vs. numerisch per CV: Unterschied ±0.002 Balanced Accuracy, also im Rauschen → **nicht sinnvoll**; das finale Feature-Set nutzt plain `Month`.

**Stratifizierung:** Kombinierter Key `Crop|Stage` (z. B. `corn|Critical`) statt nur `Crop`, für Split **und** Cross-Validation-Folds. 23 nicht-leere Kombinationen, kleinste (`cotton|Harvest`) mit 11 Zeilen — ausreichend für 70/30-Split und 5-fach-CV.

**Split-Verhältnis:** 70/30, 80/20 und 90/10 liefern vergleichbare Ergebnisse (Balanced Accuracy im Band 0.78–0.81 für beide Zielgrößen, kein klarer Trend). Wir bleiben bei 70/30 für die Vergleichbarkeit mit M1.

## Modell- und Validierungsansatz

5-fache `StratifiedKFold`-Cross-Validation (auf `Crop|Stage`) ersetzt den reinen Train/Test-Split als primäre Validierung. Verglichen wurden `RandomForestClassifier`, `HistGradientBoostingClassifier` und `MLPClassifier` auf Rohbändern bzw. Rohbändern + Feature Engineering, inkl. Train-vs-CV-Lücke als Overfitting-Check. Das beste Modell/Feature-Set pro Zielgröße wurde per `RandomizedSearchCV` nachjustiert und abschließend einmalig auf dem nie zuvor verwendeten 30 %-Testsplit ausgewertet.

**CV-Ergebnisse (Balanced Accuracy):**

| Modell | Feature-Set | Crop | Stage | Gap Train−CV (Crop/Stage) |
|---|---|---|---|---|
| RandomForest | Rohbänder | 0.805 | 0.768 | 0.20 / 0.23 |
| RandomForest | + FE | 0.841 | 0.797 | 0.16 / 0.20 |
| HistGradientBoosting | Rohbänder | 0.883 | 0.852 | 0.12 / 0.15 |
| HistGradientBoosting | + FE | **0.897** | **0.883** | 0.10 / 0.12 |
| MLP | + FE | 0.884 | 0.829 | 0.04 / 0.04 |

- **Modellwahl** ist der größte Hebel (HGB vs. RF ca. +0.08), **Feature Engineering** der zweitgrößte (HGB Stage +0.031, Crop +0.014, über der CV-Streuung).
- Bäume haben eine deutliche Train-CV-Lücke, der CV-Score ist aber hoch und stabil (Std ≤ 0.03); FE verkleinert die Lücke leicht.

**Strategievergleich (CV, Out-of-fold, gleiches MLP auf `+FE`):**

| Strategie | Crop | Stage |
|---|---|---|
| separate MLPs | 0.884 | 0.829 |
| gemeinsames MLP (Crop\|Stage, 23 Klassen) | **0.909** | 0.861 |
| separates HGB (Referenz) | 0.897 | **0.883** |

Das gemeinsame Label ist klar besser als separate MLPs: Crop- und Stage-Information stützen sich gegenseitig. Für Crop ist es in der CV sogar das beste Verfahren, für Stage bleibt separates HGB vorne.

## Ergebnisse (Testsplit, 30 %)

| Zielgröße | Modell | Balanced Accuracy | Macro F1 |
|---|---|---|---|
| Crop (M1) | RandomForest | 0.778 | 0.810 |
| Crop (M2 ohne FE) | HistGradientBoosting, Rohbänder | 0.877 | 0.882 |
| Crop (M2 final) | HistGradientBoosting + FE, getunt | **0.889** | 0.886 |
| Crop (Joint) | gemeinsames MLP + FE | **0.893** | **0.889** |
| Stage (M1) | RandomForest | 0.769 | 0.788 |
| Stage (M2 ohne FE) | HistGradientBoosting, Rohbänder | 0.874 | 0.877 |
| Stage (M2 final) | HistGradientBoosting + FE, getunt | **0.888** | **0.891** |
| Stage (Joint) | gemeinsames MLP + FE | 0.874 | 0.873 |

Verbesserung ggü. Baseline: Crop +0.111, Stage +0.119 Balanced Accuracy. Der Unterschied Joint-MLP vs. HGB bei Crop (0.004) liegt im Rauschen.

## Fehleranalyse (Confusion Matrix, siehe Notebook)
- **Crop:** `rice` (kleinste Klasse, 28 Testfälle) am schwächsten (F1 0.82, bei 28 Fällen ändern ein bis zwei Fälle den Wert um ±0.04), `corn` am stärksten (F1 0.94).
- **Stage:** `Mature_Senesc` am schwächsten, mit FE aber verbessert (F1 0.81 → 0.83, Verwechslung mit benachbarten Reifephasen); `Late` am stärksten (F1 0.96).

## Limitationen & nächste Schritte (Meilenstein 3)
- Hierarchische Klassifikation (erst Crop, dann Stage je Crop) und ein gemeinsames HGB sind noch nicht verglichen.
- Kein Feature-Selection/PCA — relevant für Meilenstein 3 (Dateninput minimieren); die Indizes bieten sich als kompakte Eingabe an.
- `rice` und `Harvest`-bezogene Kombinationen bleiben datenarm; Resampling-Strategien (z. B. SMOTE) wurden nicht getestet, nur `class_weight` bei RandomForest.
