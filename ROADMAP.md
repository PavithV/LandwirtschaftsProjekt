# Roadmap & Handoff — Domänenprojekt 2: Hyperspektral-Klassifikation

_Stand: 2026-09-29_

## Projektüberblick
Klassifikation landwirtschaftlicher Kulturpflanzen und ihrer Wachstumsstadien anhand hyperspektraler Satellitendaten (EO-1 Hyperion). Zweistufige Aufgabe:
1. Crop-Art bestimmen (corn, soybean, winter_wheat, cotton, rice)
2. Innerhalb der Crop-Klasse das Growing Stage bestimmen (Critical, Emerge_VEarly, Mature_Senesc, Early_Mid, Late, Harvest)

Quelle der Aufgabenstellung: `Beschreibung_Domaenen_Projekt_2_ Vorabversion_2026_27(1).docx` im Repo-Root.

## Aktueller Stand (2026-09-29)
- [x] Repo initialisiert und mit GitHub verknüpft (`https://github.com/PavithV/LandwirtschaftsProjekt.git`, Branch `main`)
- [x] Rohdaten committet: `train.csv`, `test.csv`, Aufgaben-docx
- [x] Aufgabenstellung gelesen und verstanden
- [x] Grobe Struktur der CSVs gesichtet (Shape, Spalten, Klassenverteilung — siehe unten)
- [x] Vertiefte explorative Datenanalyse (`eda.ipynb`)
- [x] Data-Preparation-Pipeline (Preprocessing-Entscheidungen in `baseline_model.ipynb` / `MILESTONE1_REPORT.md`)
- [x] Baseline-Modell (`baseline_model.ipynb` — RandomForest, Crop Balanced Accuracy 0.778, Stage 0.769)
- [x] Kurzreport Problemverständnis (`MILESTONE1_REPORT.md`)

## Datensatz-Fakten
- `train.csv`: 5591 Zeilen × 204 Spalten — `AEZ`, `Month`, `Crop`, `Stage`, ~198 Spektralband-Features (`X427`…`X2395`), `id`
- `test.csv`: 1398 Zeilen, gleiche Feature-Spalten, **ohne** `Crop`/`Stage` (Labels werden später zur Bewertung nachgereicht)
- Crop-Verteilung (train): corn 2097, soybean 1669, winter_wheat 1073, cotton 659, rice 93 → deutlich unausgeglichen
- Stage-Verteilung (train): Critical 1541, Emerge_VEarly 1038, Mature_Senesc 989, Early_Mid 982, Late 861, Harvest 180
- Im Sample bereits leere Felder bei einzelnen Spektralbändern beobachtet → fehlende Werte prüfen

## Meilenstein-Roadmap (aus der docx)

| Status | Datum | Meilenstein | Inhalt |
|--------|-------|-------------|--------|
| ✅ | 28.09.2026 | Kickoff Workshop | Aufgabenstellung, Bewertungsrahmen, Datenübersicht, Start Team-Workflow |
| 🔶 | 08.10.2026 | Meilenstein 1 — Domain & Baseline | Kurzreport Problemverständnis/Data Understanding, erste Preprocessing-Pipeline, erster Baseline-Klassifikator (70/30 Split), Code + Demo-Run — **Artefakte fertig** (`eda.ipynb`, `baseline_model.ipynb`, `MILESTONE1_REPORT.md`), Abgabe/Review beim Dozenten steht noch aus |
| ⬜ | 15.10.2026 | Meilenstein 2 — Modellreife & Validierung | Optimierte Preprocessing-Pipeline, verbessertes Modell mit sauberer Validierung, Vergleich ggü. Baseline, Fehleranalyse (Confusion Matrix pro Klasse) |
| ⬜ | 22.10.2026 | Meilenstein 3 — Dateninput minimieren | Studie zur Datenreduktion inkl. Trade-off-Analyse "Qualität vs. Input"; Bonus: Feature-Importance/Explainability (SHAP o.ä.) |
| ⬜ | 31.10.2026 | Abgabe | Reproduzierbarer Code (README, requirements), Abschlussbericht, Modellkarte, Vorhersage-CSV auf Validierungsdaten, Arbeitszeit-Doku, Team-Beitrags-Zusammenfassung |
| ⬜ | November | Abschlusspräsentation & Einzelgespräche | 10–15 Min Pitch + Q&A, danach Einzelreflexion |

## Nächste Schritte (Fokus: Meilenstein 2, 15.10.2026)
1. Modellierungsstrategien vergleichen: hierarchisch vs. gemeinsame Zielklasse vs. separate Modelle (Baseline nutzt separate Modelle) — siehe offene Entscheidungen
2. Preprocessing-Pipeline optimieren (Hyperparameter-Tuning, Cross-Validation statt einfachem Split)
3. Verbessertes Modell mit sauberer Validierung, Vergleich ggü. Baseline (Balanced Accuracy 0.778/0.769, siehe `MILESTONE1_REPORT.md`)
4. Detaillierte Fehleranalyse (Confusion Matrix pro Klasse, welche Klassen werden verwechselt und warum)

## Offene Entscheidungen
- Klassifikationsstrategie: hierarchisch (erst Crop, dann Stage) vs. gemeinsame Zielklasse (z. B. `corn_early`) vs. zwei separate Modelle
- Umgang mit Klassenungleichgewicht (v. a. rice stark unterrepräsentiert)
- Tabellarischer Ansatz (RF/XGBoost/SVM/MLP) vs. sequenzieller Ansatz (Spektrum als Sequenz, z. B. 1D-CNN/Transformer) — oder beides vergleichen
