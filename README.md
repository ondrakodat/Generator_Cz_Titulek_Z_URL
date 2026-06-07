# Radha Subtitle Tool

Jednoducha webova aplikace pro automaticke vytvoreni ceskych titulku a hotovych videi k serialu Radha a Krishna.

## Bezna webova obsluha

1. Spustte aplikaci.
2. Vlozte odkazy na videa, kazdy na novy radek.
3. Kliknete na **Spustit zpracovani**.
4. Pockejte na **100 %**.
5. Stahnete hotove video.

To je vse. Bezny uzivatel neresi OCR engine, crop, preset, sample rate ani styl titulku. Web pouziva maximalni automaticky kvalitni rezim pro ruske vypalene titulky a ve vychozim nastaveni vytvori hotove video s ceskymi titulky.

Volitelne lze zaskrtnout **Pouze vytvorit SRT bez videa**, pokud chcete jen soubory titulku.

## Co aplikace automaticky dela

- stahne kazde video,
- ulozi zdrojove video jako `source.mp4` v adresari jobu,
- najde oblast ruskch vypalenych titulku,
- overi prvni subtitle framy a vybere nejlepsi OCR crop,
- porovna recall/balanced varianty podle poctu platnych ruskch radku, ne podle nejmensiho sumu,
- precte ruske titulky primarne pres PaddleOCR s Tesseract fallbackem,
- rekonstruuje puvodni rusky SRT bez prekladu,
- provede konzervativni hindu context correction bez zmeny casovani a poctu bloku,
- pri podezrele malem poctu titulku automaticky zopakuje OCR s recall presetem a sirsim cropem,
- pri timeoutu jednoho OCR framu frame preskoci a pokracuje,
- uklada OCR cache pro opakovane zpracovani stejneho videa,
- prelozi do cestiny az z `subtitles_original_context_fixed.srt`,
- vypali ceske titulky do videa,
- po uspesnem vytvoreni videa smaze docasne soubory.

## Pipeline titulku

```text
Video / URL
-> output/jobs/<job_id>/source.mp4
-> detekce oblasti titulků
-> detekce framů s titulky
-> OCR maximum recall
-> hlasování a rekonstrukce bloků
-> subtitles_original_raw.srt
-> hindu context correction
-> subtitles_original_context_fixed.srt
-> překlad do češtiny
-> subtitles_cs.srt
-> volitelný burn-in do video_cz.mp4
```

## Vystupy

Kazde video ma vlastni slozku:

```text
output/jobs/job_001/
output/jobs/job_002/
output/jobs/job_003/
```

Po uspesnem vytvoreni videa zustavaji hlavne uzitecne soubory:

- `video_cz.mp4`
- `subtitles_original_raw.srt`
- `subtitles_original_context_fixed.srt`
- `subtitles_cs.srt`
- `context_corrections.json`
- `translation_report.json`
- `ocr_report.txt`
- `log.txt`

## Vychozi automaticky kvalitni rezim

Web automaticky pouziva:

- `ocr_engine = paddle`
- `subtitle_detect_mode = white-text`
- `sample_rate = 0.5`
- `ocr_min_gap = 2.0`
- `ocr_preprocess = auto`
- `ocr_preset = maximum_recall`
- `translator = none`
- `ai_cleanup = auto`
- `cleanup_temp = true`
- `burn = true`
- maximum-recall OCR oblast `x1=0.02`, `x2=0.98`, `y1=0.62`, `y2=0.98`
- automatickou validaci variant:
  - maximum: `maximum_recall`, `x=0.02-0.98`, `y=0.62-0.98`
  - A: `recall`, `x=0.03-0.97`, `y=0.68-0.96`
  - B: `recall`, `x=0.05-0.95`, `y=0.70-0.96`
  - C: `balanced`, `x=0.05-0.95`, `y=0.72-0.96`
- quality guard: pokud 50-70 minutova epizoda vyjde pod 450 bloku, OCR se zopakuje s recall presetem
- cleanup guard: pokud cleanup smaze vic nez 25 % bloku, preklada se bezpecnejsi pred-cleanup sada
- OCR report obsahuje candidates, accepted/rejected, timeout/failed/successful frames a confidence statistiky

PaddleOCR je vychozi OCR engine. Pokud neni dostupny nebo pri framu selze, aplikace pokracuje pres Tesseract fallback.

## Expertni rezim

Rozbalovaci sekce **Expertni rezim pro ladeni** je schovana ve vychozim stavu. Slouzi jen pro diagnostiku a testy.

Najdete tam napriklad:

- cookies z prohlizece,
- volbu OCR engine,
- povoleni PaddleOCR,
- testovaci casovy rozsah,
- crop hodnoty,
- debug OCR cropy,
- debug detekce titulku,
- prekladac a AI cleanup.

Debug preview cropy a detekcni obrazky se ukladaji jen pri zapnutem expert debug rezimu.

## Spuštění jedním kliknutím

Launcher pro Windows nevyzaduje Ollama. Spousti pouze Docker Desktop, Docker Compose a webove rozhrani aplikace.

Jednou vytvorte zastupce na plose:

```powershell
powershell -ExecutionPolicy Bypass -File create_desktop_shortcut.ps1
```

Na plose vznikne ikona **Radha Subtitle Tool**.

Po kliknuti na ikonu skript:

- zkontroluje prikaz `docker`,
- spusti Docker Desktop, pokud jeste nebezi,
- pocka, dokud nebude Docker dostupny,
- spusti aplikaci pres `docker compose up -d --build`,
- otevre web v prohlizeci.

Web se otevre na:

```text
http://localhost:5000
```

Pokud Docker Desktop neni nainstalovany nebo nejde spustit, skript vypise srozumitelnou chybu.

Bez zastupce lze aplikaci spustit stejnym launcherem:

```powershell
powershell -ExecutionPolicy Bypass -File start_app.ps1
```

## Spusteni bez ikony

```powershell
python web_app.py
```

Pokud lokalni Python nema zavislosti, spustte nejdriv:

```powershell
python -m pip install -r requirements.txt
```

## Docker

Spusteni webove aplikace:

```powershell
docker compose up --build
```

Otevrete:

```text
http://127.0.0.1:5000
```

CLI zustava dostupne:

```powershell
docker compose run --rm radha-tool python main.py --help
```

## Benchmark kratkeho useku

Pro porovnani presetů na kratkem useku pouzijte `--benchmark` se zacatkem a koncem:

```powershell
python main.py --input "input/video.mp4" --benchmark --start-time 00:10:00 --end-time 00:15:00 --disable-paddle --ocr-engine tesseract --subtitle-detect-mode white-text
```

Benchmark vytvori:

- `output/recall.srt`
- `output/maximum_recall.srt`
- `output/balanced.srt`
- `output/strict.srt`
- `output/benchmark_report.json`
- `output/benchmark_report.html`

Report obsahuje OCR candidates, OCR hits, pocet subtitle bloku, prumernou/median confidence, timeout/failed/successful frame counters a rychlost zpracovani. Maximum Recall by mel mit nejvyssi coverage.

Bezpecny CLI priklad s Tesseractem:

```powershell
docker compose run --rm radha-tool python main.py --input "input/video.mp4" --only-srt --disable-paddle --ocr-engine tesseract --subtitle-detect-mode white-text --ocr-preprocess auto --ocr-preset maximum_recall --translator none
```

## Poznamka

Videa se zpracovavaji postupne. Je to zamer, protoze OCR, preklad a vytvareni videa jsou narocne na vykon pocitace.
