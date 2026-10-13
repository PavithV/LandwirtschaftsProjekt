# Meilenstein 2 — Modellreife & Validierung

_Stand: 2026-10-13_

## Ausgangspunkt: Feedback zu Meilenstein 1
- Stratifizierung soll auch `Stage` berücksichtigen, nicht nur `Crop`.
- Zyklisches Encoding von `Month` prüfen.
- Weitere Modelle testen (Gradient Boosting Trees u. a.).
- `SimpleImputer` nicht mit Median, sondern mit den benachbarten Werten.
- Andere Split-Verhältnisse als 70/30 ausprobieren.
- Cross-Validation / saubere Validierung statt einfachem Train/Test-Split.

Details und Code: `milestone2_model.ipynb` (Laufzeit komplett ca. 35 Min. auf 4 Kernen; die Hyperparameter-Suche dominiert).

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

**Split-Verhältnis:** 70/30, 80/20 und 90/10 liefern vergleichbare Ergebnisse (Balanced Accuracy im Band 0.81–0.84 für beide Zielgrößen, nur leichter Anstieg mit mehr Trainingsdaten). Wir bleiben bei 70/30 für die Vergleichbarkeit mit M1.

## Modell- und Validierungsansatz

5-fache `StratifiedKFold`-Cross-Validation (auf `Crop|Stage`) ersetzt den reinen Train/Test-Split als primäre Validierung. Ablauf:
1. **Vorauswahl mit Default-Parametern:** `RandomForestClassifier`, `HistGradientBoostingClassifier` und `MLPClassifier` auf Rohbändern bzw. Rohbändern + Feature Engineering, inkl. Train-vs-CV-Lücke als Overfitting-Check.
2. **Tuning mit gleichem Budget:** HGB und MLP werden per `RandomizedSearchCV` mit jeweils **30 zufälligen Kombinationen** (× 5 Folds) getunt — für Crop, Stage und (MLP) für das gemeinsame Label `Crop|Stage`.
3. **Auswahl per CV:** Pro Zielgröße gewinnt der Kandidat mit dem höchsten getunten CV-Score; erst dann einmalige Auswertung auf dem nie zuvor verwendeten 30 %-Testsplit.

**Vorauswahl, CV mit Default-Parametern (Balanced Accuracy):**

| Modell | Feature-Set | Crop | Stage | Gap Train−CV (Crop/Stage) |
|---|---|---|---|---|
| RandomForest | Rohbänder | 0.827 | 0.793 | 0.17 / 0.21 |
| RandomForest | + FE | 0.872 | 0.829 | 0.13 / 0.17 |
| HistGradientBoosting | Rohbänder | 0.877 | 0.854 | 0.12 / 0.15 |
| HistGradientBoosting | + FE | **0.899** | **0.885** | 0.10 / 0.12 |
| MLP | + FE | 0.884 | 0.829 | 0.04 / 0.04 |

- **Feature Engineering** bringt bei allen Modellen messbar etwas (HGB Crop +0.022, Stage +0.031; RF +0.04), über der CV-Streuung.
- Mit Default-Parametern liegt HGB vorne — das ist aber **kein fairer Vergleich**, weil das MLP viel stärker von seinen Hyperparametern abhängt.

**Nach Tuning mit gleichem Budget (+FE, CV bzw. Testsplit):**

| Modell | Strategie | Crop CV | Crop Test | Stage CV | Stage Test |
|---|---|---|---|---|---|
| HistGradientBoosting | separat | 0.911 | 0.901 | 0.891 | 0.896 |
| MLP | separat | **0.932** | 0.912 | 0.907 | **0.906** |
| MLP | gemeinsames Label (23 Klassen) | 0.929 | **0.924** | **0.910** | 0.902 |
| HistGradientBoosting | gemeinsames Label (Default) | 0.550 | – | 0.550 | – |

- **Getunt schlägt das MLP das HGB** bei beiden Zielgrößen um ~0.01–0.02, in CV *und* Test. Das MLP gewinnt durch Tuning +0.05 (Crop) bzw. +0.08 (Stage), HGB nur ~+0.01.
- Entscheidend beim MLP: `early_stopping=False`, 600 Epochen, kleine Lernrate (3e-4), breite Schicht(en) mit 512 Neuronen. Das Default-`early_stopping` bricht zu früh ab.
- **Separat vs. gemeinsames Label:** getunt praktisch gleichauf (Unterschiede im Rauschen, Vorzeichen wechselt zwischen CV und Test). Der frühere Vorsprung des gemeinsamen Labels (0.909 vs. 0.884) lag an schlecht eingestellten MLPs. Vorteil des gemeinsamen Modells: ein Modell, immer konsistente Crop/Stage-Kombinationen.
- **Gemeinsames HGB** ist instabil: in 3 von 5 Folds ~0.3, in den anderen ~0.9 (vermutlich numerische Probleme bei 23 teils winzigen Klassen) — nicht weiter verfolgt.

## Ergebnisse (Testsplit, 30 %)

| Zielgröße | Modell | Balanced Accuracy | Macro F1 |
|---|---|---|---|
| Crop (M1) | RandomForest | 0.778 | 0.810 |
| Crop (M2 ohne FE) | HistGradientBoosting, Rohbänder | 0.877 | 0.882 |
| Crop (M2 final) | MLP separat + FE, getunt | **0.912** | **0.915** |
| Stage (M1) | RandomForest | 0.769 | 0.788 |
| Stage (M2 ohne FE) | HistGradientBoosting, Rohbänder | 0.874 | 0.877 |
| Stage (M2 final) | MLP gemeinsames Label + FE, getunt | **0.902** | **0.906** |

Verbesserung ggü. Baseline: Crop +0.134, Stage +0.133 Balanced Accuracy. Die finalen Modelle wurden allein per CV ausgewählt; die Testwerte der anderen Kandidaten stehen nur zur Einordnung in der Tabelle oben.

## Fehleranalyse (Confusion Matrix, siehe Notebook)
- **Crop:** `cotton` am schwächsten (F1 0.89); `rice` deutlich verbessert (F1 0.82 → 0.93, aber nur 28 Testfälle); `corn` am stärksten (F1 0.95).
- **Stage:** `Harvest` (F1 0.86, 54 Testfälle) und `Mature_Senesc` (F1 0.87) am schwächsten — benachbarte Reifephasen; `Late` am stärksten (F1 0.97).

## Limitationen & nächste Schritte (Meilenstein 3)
- Hierarchische Klassifikation (erst Crop, dann Stage je Crop) ist noch nicht verglichen; das gemeinsame HGB müsste stabilisiert werden.
- Nur ein Random Seed: MLP-Ergebnisse schwanken je nach Seed um ~0.01–0.02 — für eine belastbare Aussage MLP vs. HGB mehrere Seeds mitteln.
- Mit anderen scikit-learn-Versionen weichen v. a. die RandomForest-Werte leicht ab.
- Kein Feature-Selection/PCA — relevant für Meilenstein 3 (Dateninput minimieren); die Indizes bieten sich als kompakte Eingabe an.
- `rice` und `Harvest`-bezogene Kombinationen bleiben datenarm; Resampling-Strategien (z. B. SMOTE) wurden nicht getestet, nur `class_weight` bei RandomForest.
