# Meilenstein 2 — Modellreife & Validierung

_Stand: 2026-10-07_

## Ausgangspunkt: Feedback zu Meilenstein 1
- Stratifizierung soll auch `Stage` berücksichtigen, nicht nur `Crop`.
- Zyklisches Encoding von `Month` prüfen.
- Weitere Modelle testen (Gradient Boosting Trees u. a.).
- `SimpleImputer` nicht mit Median, sondern mit den benachbarten Werten.
- Andere Split-Verhältnisse als 70/30 ausprobieren.
- Cross-Validation / saubere Validierung statt einfachem Train/Test-Split.

Details und Code: `milestone2_model.ipynb`.

## Preprocessing-Entscheidungen (neu ggü. M1)

**Spektral-Imputation:** Die 67 komplett leeren Spektralband-Spalten werden weiterhin verworfen. Für die verbleibenden 6 Spalten mit vereinzelten Lücken (max. 24 von 5591 Zeilen) ersetzen wir die globale Median-Imputation durch **zeilenweise lineare Interpolation entlang der Wellenlänge**: Da die Spaltennamen (`X<nm>`) die Wellenlänge kodieren und Reflektanzspektren glatt verlaufen, lässt sich eine fehlende Bande besser aus den benachbarten Bändern *derselben Messung* schätzen als aus dem Median über alle Messungen. Die Interpolation ist zeilenlokal → kein Leakage zwischen Train/Test. Ein `SimpleImputer(median)` bleibt als Absicherung in der Pipeline für etwaige Randfälle.

**Month-Encoding:** `Month` reicht nur von 5 (Mai) bis 10 (Oktober) — kein Jahreswechsel, den eine zyklische Kodierung auflösen müsste. Wir haben trotzdem sin/cos-Encoding gegen die reine numerische Variante per Cross-Validation verglichen (siehe Ergebnisse): der Unterschied liegt bei ±0.002 Balanced Accuracy, also im Rauschen — zyklisches Encoding ist hier **nicht sinnvoll**, hilft aber auch nicht.

**Stratifizierung:** Kombinierter Key `Crop_Stage` (z. B. `corn_Critical`) statt nur `Crop`, für Split **und** Cross-Validation-Folds. 23 nicht-leere Kombinationen, kleinste (`cotton_Harvest`) mit 11 Zeilen — ausreichend für 70/30-Split und 5-fach-CV.

**Split-Verhältnis:** 70/30, 80/20 und 90/10 liefern vergleichbare Ergebnisse (Balanced Accuracy im Band 0.78–0.81 für beide Zielgrößen, kein klarer Trend). Wir bleiben bei 70/30 für die Vergleichbarkeit mit M1.

## Modell- und Validierungsansatz

5-fache `StratifiedKFold`-Cross-Validation (auf `Crop_Stage`) ersetzt den reinen Train/Test-Split als primäre Validierung. Verglichen wurden `RandomForestClassifier` (Baseline-Konfiguration) und `HistGradientBoostingClassifier` (Gradient-Boosting-Variante aus scikit-learn, keine neue Abhängigkeit nötig), jeweils für Crop und Stage getrennt. Das jeweils beste Modell/Encoding pro Zielgröße wurde per `RandomizedSearchCV` (kleines Suchraster) leicht nachjustiert und abschließend einmalig auf dem nie zuvor verwendeten 30 %-Testsplit ausgewertet.

**Ergebnis des Modellvergleichs:** `HistGradientBoostingClassifier` schlägt `RandomForestClassifier` deutlich (CV Balanced Accuracy Crop 0.88 vs. 0.81, Stage 0.85 vs. 0.78) — der Modellwechsel ist der größte Hebel in diesem Vergleich, deutlich größer als die Preprocessing-Anpassungen.

## Ergebnisse (Testsplit, 30 %)

| Zielgröße | Modell | Balanced Accuracy | Macro F1 |
|---|---|---|---|
| Crop (M1) | RandomForest | 0.778 | 0.810 |
| Crop (M2) | HistGradientBoosting, plain Month | **0.877** | **0.882** |
| Stage (M1) | RandomForest | 0.769 | 0.788 |
| Stage (M2) | HistGradientBoosting, zyklisches Month | **0.874** | **0.877** |

Verbesserung ggü. Baseline: Crop +0.099, Stage +0.105 Balanced Accuracy.

## Fehleranalyse (Confusion Matrix, siehe Notebook)
- **Crop:** `rice` (kleinste Klasse, 28 Testfälle) am schwächsten (F1 0.85), `corn`/`soybean` am stärksten.
- **Stage:** `Mature_Senesc` am schwächsten (F1 0.81, Verwechslung mit benachbarten Reifephasen), `Late` am stärksten (F1 0.96).

## Limitationen & nächste Schritte (Meilenstein 3)
- Kein Vergleich hierarchischer/gemeinsamer Klassifikationsstrategien — weiterhin zwei separate Modelle für Crop und Stage.
- Kein Feature-Selection/PCA — relevant für Meilenstein 3 (Dateninput minimieren).
- `rice` und `Harvest`-bezogene Kombinationen bleiben datenarm; Resampling-Strategien (z. B. SMOTE) wurden nicht getestet, nur `class_weight` bei RandomForest.
