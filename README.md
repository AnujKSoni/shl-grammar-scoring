# Grammar Scoring Engine for Spoken English

SHL Hiring Assessment 2026 (Kaggle): predict a 0–5 grammar score for a 45–60 s recording of spoken English.

**Result:** public leaderboard **0.3465** (Pearson + RMSE metric, lower is better); 5-fold speaker-grouped CV RMSE **0.532**, Pearson **0.851**.

## Files

| file | what it is |
|---|---|
| `shl_grammar_scoring.ipynb` | the full pipeline and report: data checks, features, models, results, ablation, submission |
| `submission.csv` | the submitted test predictions (216 clips) |

The competition audio and CSVs are not included: the competition rules do not allow sharing the dataset.

## Approach in one paragraph

The data needed fixing before the model mattered. The 37 clips labelled 0 are synthetic noise, so they are removed. Test clips are shorter than training clips, so every training clip is cut into four test-length pieces. Many training clips share a speaker while test speakers are new, so CV keeps every speaker inside one fold (ECAPA voice embeddings). On top of that: hand-made audio, fluency and grammar features (CoLA acceptability, grammar-correction edit rate, ERRANT error types), frozen Whisper and WavLM encoder embeddings, and DeBERTa-v3 models fine-tuned on Whisper `large-v3-turbo` transcripts. A non-negative linear stacker combines the out-of-fold predictions of about a dozen models.

The biggest single gain came from the fine-tuned transcript models (leaderboard 0.390 → 0.350); the notebook's report lists every version and the ideas that did not help.

## How to run

On Kaggle: GPU T4, Internet on, add the competition data, Run All. A first run takes about 5 hours (transcription, encoders, fine-tuning); slow steps are cached in `/kaggle/working`, so later runs take about 1.5 hours. `RUN_DEEP = False` gives a quick CPU-only baseline.
