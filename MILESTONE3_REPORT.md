# Meilenstein 3 — Dateninput minimieren

_Stand: 2026-10-16_

Details und Code: `milestone3.ipynb`. Das Notebook ist die Fortsetzung von `milestone2_model_new.ipynb` (SVM-Notebook aus M2) und läuft komplett in ca. 15 Min. auf 8 Kernen. Ausgeführt mit Python 3.14, scikit-learn 1.9.1, shap 0.50. Voraussetzung ist `data/processed/cleaned_data.csv`, erzeugt mit `python pipeline/_run_pipeline.py`.

## Fragestellung und Vorgehen

Aufgabe: Studie zur Datenreduktion inkl. Trade-off-Analyse „Qualität vs. Input" sowie, als Bonus, Feature-Importance/Erklärbarkeit. Die Beispiele der Aufgabenstellung (Sequenzen, Sommerfenster, Zeitstempel) stammen aus einem Zeitreihen-Setting. Bei uns ist jede Zeile eine einzelne Aufnahme, die Sequenzachse ist die Wellenlänge. Wir untersuchen deshalb drei Input-Achsen:
1. **Spektral:** Wie viele und welche Bänder müssen gemessen werden?
2. **Zeitlich:** Welcher Aufnahmemonat bzw. welches Fenster genügt? Wie wichtig ist `Month`?
3. **Trainingsdaten:** Wie viele gelabelte Samples sind nötig?

**Festes Modell:** die M2-Gewinner mit ihren getunten Hyperparametern — Crop: SVM separat (`C=138`, `gamma=0.0014`), Stage: SVM auf dem gemeinsamen Label `Crop|Stage` (`C=785`, `gamma=0.0023`). Gleicher 70/30-Split und gleiche 5 CV-Folds (stratifiziert auf `Crop|Stage`) wie in M2. Damit misst die Studie die Wirkung des Inputs und nicht die des Tunings. `gamma` wird mit `164 / n_features` skaliert, damit der RBF-Kernel bei weniger Features nicht breiter wird. Auswahl- und Reduktionsschritte (`SelectKBest`, PCA) laufen innerhalb der Pipeline pro Fold, es gibt also kein Leakage. Der Testsplit wird nur für das ausgewählte reduzierte Modell und die Feature-Importance genutzt.

**Wichtige Unterscheidung:** Weniger *Features* ist nicht dasselbe wie weniger *Messung*. PCA und Spektral-Statistiken brauchen weiterhin alle Bänder. Für die Datenreduktion zählt die Zahl der Bänder, die tatsächlich gemessen werden müssen. `Month`/`AEZ` sind Metadaten und praktisch kostenlos.

## Ergebnisse: spektrale Reduktion (5-fach CV, Balanced Accuracy)

| Input (jeweils + Month/AEZ) | gemessene Bänder | Features | Crop | Stage |
|---|---|---|---|---|
| Alle Features (M2-Referenz) | 131 | 164 | **0.932** | 0.912 |
| Rohbänder ohne Feature Engineering | 131 | 139 | 0.922 | **0.916** |
| 80 Bänder gleichmäßig | 80 | 88 | 0.925 | 0.912 |
| **40 Bänder gleichmäßig (ausgewählt)** | 40 | 48 | 0.925 | 0.914 |
| 20 Bänder gleichmäßig | 20 | 28 | 0.915 | 0.912 |
| 12 Vegetationsindizes + ihre 20 Bänder | 20 | 40 | 0.908 | 0.913 |
| 12 Vegetationsindizes | 20 | 20 | 0.900 | 0.901 |
| Sentinel-2 simuliert (11 Bänder) | 11 | 19 | 0.903 | 0.898 |
| 10 Bänder gleichmäßig | 10 | 18 | 0.891 | 0.900 |
| 10 Bänder `SelectKBest` (F-Test) | 10 | 18 | 0.815 | 0.831 |
| 10 PCA-Komponenten | 131 | 18 | 0.923 | 0.904 |
| 5 Bänder gleichmäßig | 5 | 13 | 0.859 | 0.882 |
| 3 Bänder gleichmäßig | 3 | 11 | 0.832 | 0.844 |
| nur Month/AEZ | 0 | 8 | 0.699 | 0.642 |

- **Die Kurve bleibt bis ~20 Bänder flach.** Mit 15 % der Bänder verliert Crop 0.017, Stage nichts. Unter 10 Bändern bricht die Leistung ein. Stage verkraftet die Reduktion besser als Crop.
- **Gleichmäßig verteilte Bänder schlagen `SelectKBest` bei jedem k.** Der univariate F-Test wählt Cluster redundanter Nachbarbänder (Crop, k=10: nur 600–691 nm). Entscheidend ist die Abdeckung von VIS, Red-Edge, NIR, SWIR1 und SWIR2.
- **PCA** komprimiert gut (10 Komponenten: Crop 0.923), reduziert aber nicht die Messung. Der Abfall ab 20 Komponenten ist ein Whitening-Artefakt: Der `StandardScaler` hinter der PCA skaliert auch die Rauschkomponenten hoch.
- **Vegetationsindizes:** 12 Indizes aus nur 20 Bändern erreichen 0.900 / 0.901 — kompakt und interpretierbar.
- **Sentinel-2-Simulation:** Die Hyperion-Bänder werden innerhalb der S2-Bandbreiten gemittelt (B1–B8A, B11, B12; B9/B10 liegen in den Wasserdampf-Lücken). Ergebnis −0.03 / −0.014 ggü. Hyperion. Ein frei verfügbarer Multispektralsensor erreicht also ~97–98 % der Leistung. Das ist eine Obergrenze, weil Rechteckfilter, gleiche Auflösung und gleiche Atmosphärenkorrektur angenommen werden.

## Ausgewähltes reduziertes Modell (Testsplit, 30 %)

Regel: die wenigsten gemessenen Bänder bei höchstens 0.01 Verlust in der CV für Crop **und** Stage (PCA ausgeschlossen). Das ergibt **40 gleichmäßig verteilte Bänder + Month/AEZ**.

| Modell | Bänder | Features | Crop BAcc | Crop Macro F1 | Stage BAcc | Stage Macro F1 |
|---|---|---|---|---|---|---|
| Voll (M2) | 131 | 164 | **0.922** | **0.923** | 0.908 | 0.909 |
| Reduziert | 40 | 48 | 0.906 | 0.910 | **0.916** | **0.915** |

Mit 70 % weniger Bändern liegt das Modell im Mittel gleichauf: Crop −0.016, Stage +0.008. Das Neu-Tuning des reduzierten Sets (20 Kandidaten × 5 Folds) wählt dieselben Parameter wie die gamma-Skalierung und bestätigt sie damit.

## Ergebnisse: Metadaten, Aufnahmezeitpunkt, Trainingsmenge

**Metadaten (CV):** Ohne AEZ verliert Crop 0.049 (0.883), ohne Month verliert Stage 0.031 (0.881), ohne beides −0.066 / −0.055. Month und AEZ allein erreichen bereits 0.70 / 0.64. Der Datensatz koppelt Kultur und Region bzw. Stadium und Monat stark (z. B. `rice` nur im August). Der Validierungsdatensatz hat fast dieselbe AEZ- und Monatsverteilung (Abweichung ≤ 1,5 Prozentpunkte). Für die Abgabe ist das unkritisch, für die Übertragung auf neue Regionen ein Risiko.

**Aufnahmemonat** (CV-Vorhersagen des vollen Modells, nach `Month` ausgewertet):

| Monat(e) | n | Crop | Stage |
|---|---|---|---|
| Mai | 303 | 0.791 | 0.855 |
| Juni | 410 | 0.916 | 0.805 |
| Juli | 1060 | 0.953 | 0.952 |
| August | 640 | 0.976 | 0.969 |
| September | 1151 | 0.888 | 0.916 |
| Oktober | 349 | 0.855 | 0.805 |
| Juni–August | 2110 | 0.953 | 0.897 |
| Juli–September | 2851 | 0.939 | 0.929 |
| alle | 3913 | 0.932 | 0.912 |

Hochsommer-Aufnahmen sind am zuverlässigsten. Bei Auflaufen und Ernte (Mai, Oktober) dominiert das Bodensignal. Pro Monat kommen nicht alle Klassen vor, die Werte sind daher nur eingeschränkt vergleichbar.

**Trainingsmenge (Lernkurve):** Die Kurve hat kein Plateau, sie steigt um ca. +0.01 pro +20 % Daten (20 % → 0.840 / 0.827, 80 % → 0.922 / 0.902, 100 % → 0.932 / 0.912). Weniger Trainingsdaten kosten direkt Leistung.

## Bonus: Feature-Importance / Erklärbarkeit

**Gruppierte Permutation Importance** (volle SVM, Testsplit, 5 Wiederholungen). Gruppen statt Einzelbänder, weil Nachbarbänder stark korreliert sind und bei Einzelband-Permutation füreinander einspringen.

| Gruppe | Abfall Crop | Abfall Stage |
|---|---|---|
| AEZ | **0.535** | 0.262 |
| NIR (763–1316 nm) | 0.304 | 0.346 |
| SWIR2 (2063–2345 nm) | 0.298 | 0.198 |
| Vegetationsindizes | 0.272 | 0.163 |
| SWIR1 (1518–1760 nm) | 0.260 | 0.215 |
| VIS (437–681 nm) | 0.237 | 0.192 |
| Red-Edge (691–752 nm) | 0.220 | 0.083 |
| Month | 0.104 | **0.435** |
| Statistiken & Gradienten | 0.095 | 0.097 |

**SHAP** (`TreeExplainer` auf HistGradientBoosting mit Rohbändern + Month/AEZ, Test-BAcc 0.867 / 0.875, 500 Testzeilen):
- **Pro Band ist der Red-Edge (~700–720 nm) am informativsten.** 7 Bänder tragen 2–26 % des |SHAP| einer Klasse, der Peak liegt bei ~700 nm (Stage) bzw. ~712 nm (Crop). Die Red-Edge-Position verschiebt sich mit dem Chlorophyllgehalt.
- `Late` und `Mature_Senesc` stützen sich auf Red-Edge und VIS (Chlorophyllabbau, Seneszenz) sowie SWIR2 (Trockensubstanz, Lignin/Zellulose um 2100–2300 nm).
- `rice` wird über NIR erkannt (44 % des |SHAP|; höchste NIR-Reflektanz, dichter Bestand im August), `winter_wheat`/`cotton` über AEZ (47 / 41 %), `Early_Mid` über den Monat (38 %).
- In der Permutation wirkt Red-Edge kleiner, weil dieselbe Information auch in NDRE/REP/CI und in den Nachbarregionen steckt. Ein kleiner Abfall bedeutet „ersetzbar", nicht „irrelevant".

## Fazit

- Für die Abgabe bleibt das **volle M2-Modell** die erste Wahl: beste Crop-Leistung, und die Validierungsdaten enthalten alle Bänder.
- Antwort auf die Datenreduktions-Frage: **20–40 gleichmäßig verteilte Bänder + Month/AEZ erreichen ≥ 98 % der Leistung, 11 Sentinel-2-Bänder ~97 %.** Unter ~10 Bändern bricht die Leistung ein.
- Wichtiger als einzelne trennscharfe Bänder ist die Abdeckung aller Spektralregionen. Das informativste Einzelgebiet ist der Red-Edge.
- Eine einzelne Aufnahme im Juli oder August reicht für eine zuverlässige Crop-Klassifikation.

## Limitationen & nächste Schritte (Abgabe)

- Ein Split, ein Seed. gamma wird skaliert statt pro Konfiguration getunt (Ausnahme: das ausgewählte Set).
- Die Sentinel-2-Simulation ist idealisiert. SHAP läuft auf einem HGB statt auf der SVM (KernelExplainer auf der SVM wäre zu ungenau oder zu teuer).
- Die starke Abhängigkeit von AEZ/Month ist ein Generalisierungsrisiko für Regionen außerhalb des Datensatzes.
- Nächste Schritte Richtung Abgabe: Vorhersage-CSV auf den Validierungsdaten (`test.csv` durch `pipeline/src/data_prep.py` mit den AEZ-Kategorien des Trainings), README/Repro-Pfad, Modellkarte, Abschlussbericht.
