# Roadmap & Handoff — Domänenprojekt 2: Hyperspektral-Klassifikation

_Stand: 2026-10-16_

## Projektüberblick
Klassifikation landwirtschaftlicher Kulturpflanzen und ihrer Wachstumsstadien anhand hyperspektraler Satellitendaten (EO-1 Hyperion). Zweistufige Aufgabe:
1. Crop-Art bestimmen (corn, soybean, winter_wheat, cotton, rice)
2. Innerhalb der Crop-Klasse das Growing Stage bestimmen (Critical, Emerge_VEarly, Mature_Senesc, Early_Mid, Late, Harvest)

Quelle der Aufgabenstellung: `Beschreibung_Domaenen_Projekt_2_ Vorabversion_2026_27(1).docx` im Repo-Root.

## Aktueller Stand (2026-10-16)
- [x] Meilenstein 3 (`milestone3.ipynb`, `MILESTONE3_REPORT.md`): Datenreduktions-Studie mit festem M2-SVM-Modell — Feature-Gruppen, Indizes, Bandauswahl (gleichmäßig / SelectKBest / PCA), Sentinel-2-Simulation, Aufnahmemonat, Lernkurve, Trade-off-Kurve; Bonus: gruppierte Permutation Importance + SHAP. Ergebnis: 40 gleichmäßig verteilte Bänder (−70 %) Test Crop 0.906 / Stage 0.916 (voll 0.922 / 0.908); 20 Bänder ≥ 98 % der Leistung, Sentinel-2 (11 Bänder) ~97 %; Red-Edge pro Band am informativsten; starke Abhängigkeit von AEZ/Month
- [x] Meilenstein 2 abgeschlossen: finales SVM-Notebook (jetzt Basis von `milestone3.ipynb`) — Test Crop 0.922 (SVM separat), Stage 0.908 (SVM gemeinsames Label); `MILESTONE2_REPORT.md` beschreibt noch den früheren MLP-Stand
- [x] `pipeline/` (Data Prep + EDA) und `data/` (Roh-CSVs) versioniert; `data/processed/cleaned_data.csv` wird per `python pipeline/_run_pipeline.py` erzeugt (gitignored)

## Stand 2026-10-13
- [x] Meilenstein 2 erweitert: Feature Engineering (Indizes inkl. Red-Edge/PSRI, Gradienten pro nm, Statistiken; aus `features.ipynb` übernommen/überarbeitet), MLP, Train-vs-CV-Gap, Strategievergleich separat vs. gemeinsames Label, faires Tuning (30er-Suche für HGB und MLP) — Holdout Crop 0.912 (MLP separat), Stage 0.902 (MLP gemeinsames Label); getuntes HGB Crop 0.901 / Stage 0.896
- [x] Meilenstein 2 (2026-10-07): optimierte Preprocessing-Pipeline (Spektral-Interpolation, Crop_Stage-Stratifizierung), 5-fach CV, Modellvergleich RandomForest vs. HistGradientBoosting, Fehleranalyse (`milestone2_model.ipynb`, `MILESTONE2_REPORT.md`) — Crop Balanced Accuracy 0.877 (M1: 0.778), Stage 0.874 (M1: 0.769)

## Vorheriger Stand (2026-09-29)
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
| ✅ | 15.10.2026 | Meilenstein 2 — Modellreife & Validierung | Optimierte Preprocessing-Pipeline, verbessertes Modell mit sauberer Validierung, Vergleich ggü. Baseline, Fehleranalyse (Confusion Matrix pro Klasse) — `milestone2_model.ipynb`, SVM-Notebook (Git-History), `MILESTONE2_REPORT.md` |
| 🔶 | 22.10.2026 | Meilenstein 3 — Dateninput minimieren | Studie zur Datenreduktion inkl. Trade-off-Analyse "Qualität vs. Input"; Bonus: Feature-Importance/Explainability (SHAP o.ä.) — **Artefakte fertig** (`milestone3.ipynb`, `MILESTONE3_REPORT.md`), Vorstellung steht noch aus |
| ⬜ | 31.10.2026 | Abgabe | Reproduzierbarer Code (README, requirements), Abschlussbericht, Modellkarte, Vorhersage-CSV auf Validierungsdaten, Arbeitszeit-Doku, Team-Beitrags-Zusammenfassung |
| ⬜ | November | Abschlusspräsentation & Einzelgespräche | 10–15 Min Pitch + Q&A, danach Einzelreflexion |

## Nächste Schritte (Fokus: Abgabe, 31.10.2026)
1. Vorhersage-CSV auf den Validierungsdaten: `test.csv` mit `pipeline/src/data_prep.py` aufbereiten (AEZ-Kategorien aus dem Training via `reference_path`), finale Modelle (Crop: SVM separat, Stage: SVM gemeinsames Label) auf dem gesamten Trainingsdatensatz fitten
2. README mit Repro-Pfad (Pipeline → Notebooks), `requirements.txt` mit Versionen pinnen (getestet: Python 3.14, scikit-learn 1.9.1, shap 0.50)
3. Abschlussbericht, Modellkarte (Annahmen/Grenzen, u. a. AEZ/Month-Abhängigkeit aus M3), Arbeitszeit-Doku, Team-Beitrags-Zusammenfassung
4. `MILESTONE2_REPORT.md` auf den finalen SVM-Stand bringen
5. Optional: mehrere Seeds für die Hauptvergleiche; Resampling für `rice`/`Harvest` (bisher nur `class_weight`)

## Offene Entscheidungen
- Klassifikationsstrategie: hierarchisch (erst Crop, dann Stage) vs. gemeinsame Zielklasse (z. B. `corn_early`) vs. zwei separate Modelle
- Umgang mit Klassenungleichgewicht (v. a. rice stark unterrepräsentiert)
- Tabellarischer Ansatz (RF/XGBoost/SVM/MLP) vs. sequenzieller Ansatz (Spektrum als Sequenz, z. B. 1D-CNN/Transformer) — oder beides vergleichen
