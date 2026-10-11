# Digitization notes

Decisions, oddities and findings from digitizing the SJCET register. This is the
raw material for the report's "challenges" section.

## Method used for July 2010 – December 2016 (the clean scans)

The OCR route in the guide could not be run on this machine. There is no CUDA GPU
to fine-tune TrOCR, Docker isn't reachable from WSL (so GloSAT can't run), and
zero-shot TrOCR is only ~33% exact. Instead, every page of
`About/Dataset/Main/July2010-Dec2016.pdf` was rendered at 200 dpi and **transcribed by
hand, cell by cell, exactly as written** into `work/transcripts/mNN.csv` (one file per
PDF page, `mNN` = PDF page number = `page_id`). Each file has a header line with the
source, year, month and column order.

The transcriptions were then validated automatically:

```bash
uv run python check_transcripts.py   # per-page: calendar, ranges, wet<=dry, min<max, rain sum
uv run python assemble.py            # -> ../data/sjcet/sjcet_daily.csv, work/pages.csv, work/qc_report.csv
```

The transcripts double as **OCR training labels** (guide step 6) once cells are
cropped. Every cell's true text is known, so a fine-tuned OCR model can be trained
and scored against them later.

## Rainfall checksums (`work/monthly_totals.csv`)

- **2014–2016** pages carry the month total at the foot of the rain column.
- The **summary PDF** (`Sum/July2010_Dec2016.pdf`) has a page per month with "Rainfall of
  the month / of the year" and a table of the same month in other years. The 2016
  pages list 2012–2016 and the 2011 pages list 2009–2011, which gives a written
  total for **every month, July 2010 – December 2016**.
- Result: **59 of 78 months** match the written total to within 0.1 cm.
- Where a reading was genuinely ambiguous (e.g. `8.32` vs `3.32`) and one option
  made the month match exactly, that option was chosen and the remark says so.
- For 6 of the mismatched months (Nov 2016, Aug 2016, Mar 2016, Nov 2015, Oct 2014,
  May 2014) the rain column was re-read zoomed in. The digits are as transcribed,
  so these are **the observer's addition errors**. Sep 2011 was also re-read and
  matches the page.
- In the remaining mismatched months, every rain value is kept but flagged `suspect`
  with the sums in `qc_note`. Several differ by whole numbers (+5.00, +6.00,
  +7.00, +17.00), which suggests a single misread digit, a missing page row,
  or a total copied wrongly.

## Decisions

| Situation | Stored as |
|---|---|
| Blank row (weekend/holiday/not observed) | all values empty, `qc_flag=ok`, remark copied to `qc_note` |
| Rain column blank on an observed day (2011–2013) | `rain_mm` empty, i.e. **not recorded**, not 0 |
| Rain written `0`, `0.0`, `00.0` | `0` |
| Rain written as a dash `–` | empty (ambiguous between "nil" and "not read"; noted) |
| Overwritten/struck values | the final (corrected) value, `suspect` if unclear |
| Rain on the first day after a gap (e.g. 16.2 cm after 2 blank days) | kept on the written date, `suspect`, note "may be accumulated" |
| Values that break physics as written (wet > dry, min ≥ max) | kept as written, `suspect` |
| Rain-only rows (no temperatures), e.g. Mar 2013 | rain kept, temperatures empty |
| `x.5` readings (e.g. 24.5) | kept |

Rain is converted cm → mm (×10). RH is computed from dry/wet bulb with the
Stevenson-screen psychrometer formula (guide §11). Its median is ~92% and its
range 57–100%.

## Oddities found

- **79 PDF pages for 78 months:** page 6 is a second scan of August 2016 (`m06`, duplicate of `m05`).
- **Column order changes:** 2010-07 → 2011-12 is `date|time|rain|dry|wet|min|max`;
  2012-01 onwards is `date|time|dry|wet|min|max|rain|remarks`.
- **June 2011:** from the 20th the min/max headers are swapped and relabelled (MAX|MIN).
  Values were transcribed into the correct fields.
- **September 2011:** the dates are written as `-10`, but the header says September 2011.
  Rows 20–24 were shifted one column and corrected in place.
- **June 2013:** ink from the facing page bleeds through on days 9–16. Some values are unreadable.
- **March 2013:** many days have only a rain value. A circled `2` on the 31st carries a note
  that 2 cm was "considered as from next month" (see the 1 April 2013 correction note).
- **Blank months:** November and December 2012 pages have no daily entries at all,
  yet the summary pages give totals for them (22.20 and 6.30 cm). The daily
  observations for these months are missing.
- **2011–2013 rain gaps:** many observed days have temperatures but an empty rain
  column. These are treated as not recorded.
- Max temperature is often constant for long runs (e.g. 34 °C every day in Jan 2013).
  The max thermometer was probably read or reset irregularly. Worth checking against ERA5.

## Coverage (scans)

| Year | Observed days |
|---|---|
| 2010 (Jul–Dec) | 110 |
| 2011 | 251 |
| 2012 | 198 (Nov–Dec blank) |
| 2013 | 233 |
| 2014 | 254 |
| 2015 | 250 |
| 2016 | 253 |

1,549 observed days in 2,376 calendar days (65%). Weekends and holidays make up most of the gaps.

## Not done yet

- **Phone photos** (`About/Dataset/Main/*.jpg`, 289): some repeat 2010–2016 (use the scans),
  and the rest cover **January 2017 onwards**, at least to January 2022 (a "Covid-19 closed"
  page). They need inventory (`pages.csv`), rotation/perspective correction and
  transcription the same way.
- `cells.csv` for the review tool: needs cell cropping (`grid_cells.py`) to
  attach crops to the transcribed values.
- A residual error rate: a second person should re-read a random 3–5% of `ok` cells.
