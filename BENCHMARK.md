# Benchmark Specification

This project follows a research-adapted evaluation protocol (see `ai-research-code-workflow.md`).
Part A defines metrics; Part B accumulates results. Changing Part A = MAJOR version bump.

---

# Part A — Definitions (changes require MAJOR bump)

## Evaluation Metrics

### UTMOS (Naturalness)

- **What**: No-reference naturalness MOS predictor (1–5 scale)
- **Implementation**: `torchaudio.pipelines.SQUIM_SUBJECTIVE` via `scripts/evaluate_quality.py`
- **Reduction**: Per-file score; report mean across evaluation set
- **Higher = better** (more natural / cleaner output)

### SECS (Speaker Similarity)

- **What**: Speaker embedding cosine similarity between output and reference
- **Implementation**: SpeechBrain ECAPA-TDNN embeddings, cosine distance
- **Reduction**: Per-file score; report mean across evaluation set
- **Scale**: 0–1, higher = output speaker closer to reference
- **Reference**: The same reference audio used for voice conversion

### WER (Content Preservation)

- **What**: Whisper re-transcription word error rate
- **Implementation**: Whisper-small transcribes both source and output; word-level Levenshtein / source-length
- **Reduction**: Per-file score; report mean across evaluation set
- **Scale**: 0+, lower = content better preserved (can exceed 1.0 with insertions)
- **Caveat**: TTS-clone models (text path) trivially achieve low WER; SECS/UTMOS remain meaningful

## Evaluation Protocol

- Source audio: 10s speech clip (single speaker, clean)
- Reference audio: 10–60s of target speaker
- Run `scripts/evaluate_quality.py --source X --reference Y --outputs Z1 Z2 ...`
- Report per-model numbers with code version and git commit hash
- For publication: 3+ source/reference pairs, report mean +/- std

## Audio Conditions

- Sample rate: 16kHz (resampled internally by each model)
- Format: WAV, mono, float32 or int16
- Source content: read speech (LibriSpeech-style) or conversational

---

# Part B — Results (accumulating; updated with each evaluation run)

## v0.1.x Results (current metric definitions)

### True VC Models

| Model | Code | Source | Reference | UTMOS | SECS | WER | Hardware | Notes |
|-------|------|--------|-----------|-------|------|-----|----------|-------|
| *(to be populated with first formal evaluation run)* | | | | | | | | |

### TTS-Clone Models (text path)

| Model | Code | Source | Reference | UTMOS | SECS | WER | Hardware | Notes |
|-------|------|--------|-----------|-------|------|-----|----------|-------|
| *(to be populated)* | | | | | | | | |

### Speed (RTF on Apple Silicon)

| Model | Code | Hardware | Audio Length | Wall Time | RTF | Notes |
|-------|------|----------|-------------|-----------|-----|-------|
| *(to be populated from bench_models.py runs)* | | | | | | |

---

## Abandoned Directions

*(Models or approaches tried and dropped — record here for posterity)*
