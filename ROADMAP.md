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
- [ ] Vertiefte explorative Datenanalyse
- [ ] Data-Preparation-Pipeline
- [ ] Baseline-Modell
- [ ] Kurzreport Problemverständnis (Meilenstein 1)

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
| ⬜ | 08.10.2026 | Meilenstein 1 — Domain & Baseline | Kurzreport Problemverständnis/Data Understanding, erste Preprocessing-Pipeline, erster Baseline-Klassifikator (70/30 Split), Code + Demo-Run |
| ⬜ | 15.10.2026 | Meilenstein 2 — Modellreife & Validierung | Optimierte Preprocessing-Pipeline, verbessertes Modell mit sauberer Validierung, Vergleich ggü. Baseline, Fehleranalyse (Confusion Matrix pro Klasse) |
| ⬜ | 22.10.2026 | Meilenstein 3 — Dateninput minimieren | Studie zur Datenreduktion inkl. Trade-off-Analyse "Qualität vs. Input"; Bonus: Feature-Importance/Explainability (SHAP o.ä.) |
| ⬜ | 31.10.2026 | Abgabe | Reproduzierbarer Code (README, requirements), Abschlussbericht, Modellkarte, Vorhersage-CSV auf Validierungsdaten, Arbeitszeit-Doku, Team-Beitrags-Zusammenfassung |
| ⬜ | November | Abschlusspräsentation & Einzelgespräche | 10–15 Min Pitch + Q&A, danach Einzelreflexion |

## Nächste Schritte (Fokus: Meilenstein 1, 08.10.2026)
1. Explorative Datenanalyse vertiefen: fehlende Werte, Ausreißer, Klassenüberschneidungen, spektrale Signaturen je Crop/Stage visualisieren
2. Data-Preparation-Entscheidungen treffen und dokumentieren (fehlende Werte, Normalisierung, ggf. Bandauswahl/Glättung)
3. Modellierungsansatz für die Baseline festlegen (siehe offene Entscheidungen) und einfaches Baseline-Modell mit 70/30 Split trainieren
4. Kurzreport zum Problemverständnis schreiben

## Offene Entscheidungen
- Klassifikationsstrategie: hierarchisch (erst Crop, dann Stage) vs. gemeinsame Zielklasse (z. B. `corn_early`) vs. zwei separate Modelle
- Umgang mit Klassenungleichgewicht (v. a. rice stark unterrepräsentiert)
- Tabellarischer Ansatz (RF/XGBoost/SVM/MLP) vs. sequenzieller Ansatz (Spektrum als Sequenz, z. B. 1D-CNN/Transformer) — oder beides vergleichen
