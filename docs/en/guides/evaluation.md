# Evaluation Metrics

mlx-vc ships with `scripts/evaluate_quality.py`, an objective benchmark for VC outputs. Three metrics, three different things they capture, three different ways they can mislead — read this page before drawing conclusions from the numbers.

## TL;DR

| Metric | Range | Higher / lower better | What it captures |
|--------|-------|----------------------|------------------|
| **UTMOS** | 1–5 | Higher | Naturalness / cleanliness (no-reference MOS) |
| **SECS**  | 0–1 | Higher | Speaker identity match vs reference |
| **WER**   | 0+  | Lower  | Content preservation vs source |

For voice conversion, all three matter:

- **High UTMOS, low SECS** → output is clean but doesn't sound like the target → ToneColorConverter-style failure
- **Low UTMOS, high SECS** → sounds like the target but with vocoder artifacts → kNN-VC / HiFi-GAN-style failure
- **High UTMOS, high SECS, high WER** → sounds like the target but with garbled content → diffusion model collapse

## Running the benchmark

```bash
python scripts/evaluate_quality.py \
    --source path/to/source.wav \
    --reference path/to/reference.wav \
    --outputs out_seedvc.wav out_openvoice.wav out_knnvc.wav \
    --json metrics.json
```

The script prints a 4-column table and optionally writes a JSON file with full transcripts (handy for debugging individual word errors).

It loads three models on first call:

- `openai-whisper small` (~500 MB) for WER transcription
- `speechbrain/spkrec-ecapa-voxceleb` (~25 MB) for SECS embeddings
- `torchaudio.pipelines.SQUIM_SUBJECTIVE` (~360 MB) for UTMOS

All three run on CPU.

## UTMOS (SQUIM_SUBJECTIVE)

UTMOS predicts a Mean Opinion Score (MOS) — what a human listener would rate the audio's naturalness on a 1–5 scale. We use torchaudio's SQUIM_SUBJECTIVE, which was trained to match human MOS ratings on a held-out test set.

### Algorithm

```
function UTMOS(audio, non_matching_reference):
    audio_features = HuBERT_local(audio)               # [T, 1024]
    nmr_features = HuBERT_local(non_matching_reference)
    pooled = cross_attention(audio_features, nmr_features)
    mos = MLP_regressor(pooled)                         # scalar in [1, 5]
    return mos
```

### What it's good for

- No-reference: doesn't need ground-truth audio — only one input
- Reasonably correlated with human MOS (paper reports r ~0.78)
- Sensitive to vocoder artifacts and additive noise

### What it isn't good for

- It does not measure speaker identity — high UTMOS is necessary but not sufficient for VC quality
- The "non-matching reference" is just an anchor — passing the actual reference works fine in practice but use the same anchor across runs for comparability
- Model-specific bias: SQUIM scores TTS-clone outputs lower than humans actually do (smooth synthetic speech sometimes scores 2.8 even when humans rate it 4+)

## SECS (Speaker Embedding Cosine Similarity)

SECS asks "does the output sound like the reference speaker?" by comparing fixed-dimension speaker embeddings.

### Algorithm

```
function SECS(output_audio, reference_audio):
    out_emb = ECAPA_TDNN(output_audio)         # [192]
    ref_emb = ECAPA_TDNN(reference_audio)      # [192]
    return cos(out_emb, ref_emb)               # = (out · ref) / (|out| · |ref|)
```

We use `speechbrain/spkrec-ecapa-voxceleb` — ECAPA-TDNN trained on VoxCeleb, ~7000 speakers.

### Anchors

Calibrate your expectations:

- Same speaker, two different recordings: cosine ≈ 0.7+
- Different speakers: typically < 0.5
- Same recording vs itself: 1.0
- Demucs-cleaned reference vs the original noisy reference: ≈ 0.85 (so 0.85 is roughly an upper bound for "as close as denoising would allow")

A VC output scoring 0.84 against the reference is essentially as close as the cleaned-vs-noisy of the same audio — Seed-VC reaches this in our benchmark.

### Pitfalls

- ECAPA was trained on ~3-second utterances. Very short outputs (< 1 s) give noisy SECS
- Cross-language conversions can score lower than expected even when the timbre transfer is good — ECAPA was not trained on that distribution
- TTS-clone outputs can hit very high SECS because TTS models with explicit speaker conditioning are *designed* to maximize this. SECS does not penalize the lost source prosody

## WER (Word Error Rate)

WER measures how much of the source content survived the conversion. We Whisper-transcribe both source and output, normalize, and compute word-level Levenshtein distance.

### Algorithm

```
function WER(reference_text, hypothesis_text):
    R = reference_text.split()
    H = hypothesis_text.split()
    n, m = len(R), len(H)
    DP[0..n][0..m]
    DP[i][0] = i      # deletion cost
    DP[0][j] = j      # insertion cost
    for i in 1..n:
        for j in 1..m:
            if R[i-1] == H[j-1]:
                DP[i][j] = DP[i-1][j-1]
            else:
                DP[i][j] = 1 + min(
                    DP[i-1][j-1],   # substitution
                    DP[i-1][j],     # deletion
                    DP[i][j-1],     # insertion
                )
    return DP[n][m] / n             # normalized by reference length

function VC_WER(source_audio, output_audio):
    src_text = whisper.transcribe(source_audio).text
    out_text = whisper.transcribe(output_audio).text
    return WER(normalize(src_text), normalize(out_text))
```

`normalize` lowercases, strips punctuation, and collapses whitespace.

### Range

- 0.00 = identical transcripts
- 0.25 = roughly 1 word in 4 wrong
- 1.00 = every reference word wrong (or insertions equal to reference length)
- > 1.00 possible when the hypothesis is much longer than the reference and mostly wrong (insertions push the error count above `len(R)`)

### TTS-clone caveat

TTS-clone models pipe `source → Whisper → text → TTS → output`. Whisper is then asked to transcribe the output back. Because Whisper is the same model on both ends, this is essentially a self-consistency loop and WER trivially approaches 0. **WER is not comparable between true VC and TTS-clone models** — TTS-clone wins on WER by construction, regardless of audio quality.

### Other pitfalls

- Whisper has hallucinations: silent / very-short audio can transcribe to plausible-but-wrong text
- Whisper struggles with anime / pitch-shifted voices (RVC's Zundamon scores WER 1.0 even though the audio is fine — Whisper just can't read it)
- Quiet outputs (RVC peak ~0.2 before our normalization) make Whisper drop frames

## Latest scoreboard

Source: 10-second English lecture clip. Reference: Demucs-separated 60-second clean clip.

```
Category   Model           UTMOS↑   SECS↑   WER↓   Notes
========================================================================
[anchor]   reference        3.97    1.000     —    Upper bound
true VC    seed-vc          3.97    0.847   0.06   Best overall
true VC    openvoice        3.94    0.675   0.19   Tone-color, fast
true VC    freevc-s         3.83    0.389   0.69   No speaker encoder
true VC    freevc           3.58    0.235   0.56   With speaker encoder
true VC    knn-vc           2.82    0.467   0.00   HiFi-GAN artifacts drag UTMOS
true VC    meanvc           3.99    0.282   0.88   Chinese-trained → high WER on EN
true VC    speecht5         1.28    0.182   0.94   Domain mismatch collapse
true VC    rvc              3.95    0.064   1.00   Zundamon by design
TTS-clone  cosyvoice        2.97    0.708   0.12   Goes via Whisper text
TTS-clone  pocket-tts       3.94    0.377   0.25   Lightweight TTS
```

## How to interpret a row

For a true VC model:

- **UTMOS within 0.05 of reference** ✓ output is clean
- **UTMOS more than 0.5 below reference** → vocoder artifacts or AR collapse; investigate
- **SECS ≥ 0.7** ✓ matches target speaker
- **SECS 0.4–0.7** → partial transfer; tone-color models (OpenVoice) live here
- **SECS < 0.3** → either (a) wrong speaker (RVC), (b) speaker encoder out of distribution (MeanVC on EN), or (c) model fundamentally not converting (SpeechT5 collapse)
- **WER < 0.1** ✓ content preserved
- **WER 0.1–0.4** → some drift; tone-color and one-shot VC live here
- **WER > 0.5** → content significantly degraded; check transcript to distinguish "model collapsed" from "Whisper can't read this voice"

The Zundamon RVC outlier is a useful sanity check: high UTMOS (the audio is fine), tiny SECS (it's not the reference voice), WER 1.0 (Whisper can't transcribe anime). Three different metrics catching three different facts about one output.
