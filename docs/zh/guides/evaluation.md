# 評估指標

mlx-vc 內附 `scripts/evaluate_quality.py` 跑 VC output 的客觀 benchmark。三個指標、三個不同面向、三種可能誤導 — 看這頁再下結論。

## TL;DR

| 指標 | 範圍 | 高/低為佳 | 衡量什麼 |
|------|------|-----------|----------|
| **UTMOS** | 1–5 | 高 | 自然度 / 乾淨度（no-reference MOS） |
| **SECS**  | 0–1 | 高 | 跟 reference 是否同一個 speaker |
| **WER**   | 0+  | 低 | source 內容是否保留 |

對 voice conversion 三個都重要：

- **UTMOS 高，SECS 低** → 音檔乾淨但不像 target → ToneColorConverter 路線的失敗模式
- **UTMOS 低，SECS 高** → 像 target 但有 vocoder artifacts → kNN-VC / HiFi-GAN 路線的失敗模式
- **UTMOS 高、SECS 高、WER 高** → 像 target 但內容跑掉了 → diffusion model collapse

## 跑 benchmark

```bash
python scripts/evaluate_quality.py \
    --source path/to/source.wav \
    --reference path/to/reference.wav \
    --outputs out_seedvc.wav out_openvoice.wav out_knnvc.wav \
    --json metrics.json
```

Script 印出 4 欄表格，可選擇輸出 JSON（含完整轉錄，方便 debug 個別字錯）。

第一次呼叫會載入三個模型：

- `openai-whisper small`（~500MB）做 WER 轉錄
- `speechbrain/spkrec-ecapa-voxceleb`（~25MB）算 SECS embeddings
- `torchaudio.pipelines.SQUIM_SUBJECTIVE`（~360MB）預測 UTMOS

三個都跑 CPU。

## UTMOS（SQUIM_SUBJECTIVE）

UTMOS 預測 Mean Opinion Score（MOS）— 真人聽到會打 1–5 自然度的多少分。我們用 torchaudio 的 SQUIM_SUBJECTIVE，訓練目標是 match 真人 MOS rating。

### 演算法

```
function UTMOS(audio, non_matching_reference):
    audio_features = HuBERT_local(audio)              # [T, 1024]
    nmr_features = HuBERT_local(non_matching_reference)
    pooled = cross_attention(audio_features, nmr_features)
    mos = MLP_regressor(pooled)                       # 純量 [1, 5]
    return mos
```

### 適用場合

- No-reference：不需要 ground-truth，只要一個 input
- 跟 human MOS 相關性 r ~0.78（paper 數據）
- 對 vocoder artifacts 跟 additive noise 敏感

### 不適用場合

- 不衡量 speaker identity — 高 UTMOS 是 VC 品質的必要條件不是充分條件
- "Non-matching reference" 只是錨點 — 用實際 reference 在實務上也 OK，但**整批 evaluation 用同一個錨點**才能比
- Model-specific bias：SQUIM 對 TTS-clone outputs 評分系統性偏低（合成乾淨的語音可能拿 2.8，真人會給 4+）

## SECS（Speaker Embedding Cosine Similarity）

SECS 問「output 聽起來像 reference speaker 嗎」— 透過比固定維度的 speaker embedding 算 cosine similarity。

### 演算法

```
function SECS(output_audio, reference_audio):
    out_emb = ECAPA_TDNN(output_audio)        # [192]
    ref_emb = ECAPA_TDNN(reference_audio)     # [192]
    return cos(out_emb, ref_emb)              # = (out · ref) / (|out| · |ref|)
```

我們用 `speechbrain/spkrec-ecapa-voxceleb` — VoxCeleb 訓練的 ECAPA-TDNN，~7000 speakers。

### 錨點

校準預期值：

- 同一人不同 utterance：cosine ≈ 0.7+
- 不同人：通常 < 0.5
- 同一個錄音對自己：1.0
- Demucs 清過 vs 原始 noisy reference：≈ 0.85（差不多是「降噪能拉到的上限」）

VC output 對 reference 拿 0.84，等於跟同一份音檔的 cleaned-vs-noisy 一樣近 — Seed-VC 在我們 benchmark 達到這個。

### 陷阱

- ECAPA 訓練資料是 ~3 秒 utterance。Output 過短（< 1 秒）SECS 會雜訊大
- 跨語言 conversion 即使音色 transfer 很好，SECS 可能偏低 — ECAPA 沒訓練到該分布
- TTS-clone outputs 可能拿到很高 SECS — 因為 TTS 模型有顯式 speaker conditioning 就是被訓練去最大化這個。SECS 不會懲罰失去的 source prosody

## WER（Word Error Rate）

WER 衡量 source 的內容轉換後保留多少。我們 Whisper 轉錄 source 跟 output 然後算 word-level Levenshtein 距離。

### 演算法

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
    return DP[n][m] / n             # 用 reference 長度 normalize

function VC_WER(source_audio, output_audio):
    src_text = whisper.transcribe(source_audio).text
    out_text = whisper.transcribe(output_audio).text
    return WER(normalize(src_text), normalize(out_text))
```

`normalize` 全小寫、去標點、collapse whitespace。

### 範圍

- 0.00 = 轉錄完全相同
- 0.25 = 大約 4 個字錯 1 個
- 1.00 = 每個 reference 字都錯（或 insertion 跟 reference 一樣長）
- > 1.00 可能 — hypothesis 比 reference 長很多且大部分錯時，insertion 把 error count 推過 `len(R)`

### TTS-clone 的注意事項

TTS-clone 模型走 `source → Whisper → text → TTS → output`。然後 Whisper 又被叫回去轉 output。因為兩端是同一個 Whisper，這本質上是 self-consistency loop，WER 會很容易接近 0。**WER 不能拿 true VC 跟 TTS-clone 同台比** — TTS-clone 在 WER 上 by construction 就贏，不管音質怎樣。

### 其他陷阱

- Whisper 會幻覺：靜音 / 很短的音檔可能轉成貌似合理但錯的文字
- Whisper 對動漫聲 / pitch-shifted 聲不行（RVC Zundamon 即使 audio 沒問題 WER 仍然 1.0 — Whisper 純粹聽不懂）
- 太小聲的 output（RVC normalization 前 peak ~0.2）會讓 Whisper 漏 frame

## 最新 scoreboard

Source：10 秒英文 lecture clip。Reference：Demucs 分離過的 60 秒乾淨 clip。

```
類別       Model           UTMOS↑   SECS↑   WER↓   備註
========================================================================
[anchor]   reference        3.97    1.000     —    上限
true VC    seed-vc          3.97    0.847   0.06   全方位最佳
true VC    openvoice        3.94    0.675   0.19   Tone-color，快
true VC    freevc-s         3.83    0.389   0.69   無 speaker encoder
true VC    freevc           3.58    0.235   0.56   有 speaker encoder
true VC    knn-vc           2.82    0.467   0.00   HiFi-GAN artifacts 拖低 UTMOS
true VC    meanvc           3.99    0.282   0.88   中文訓練 → 英文 WER 高
true VC    speecht5         1.28    0.182   0.94   Domain mismatch collapse
true VC    rvc              3.95    0.064   1.00   Zundamon by design
TTS-clone  cosyvoice        2.97    0.708   0.12   走 Whisper 文字路徑
TTS-clone  pocket-tts       3.94    0.377   0.25   輕量 TTS
```

## 一行怎麼讀

對真 VC 模型：

- **UTMOS 跟 reference 差 < 0.05** ✓ output 乾淨
- **UTMOS 比 reference 低 0.5 以上** → vocoder artifacts 或 AR collapse；要查
- **SECS ≥ 0.7** ✓ 像 target speaker
- **SECS 0.4–0.7** → 部分轉換；tone-color 模型（OpenVoice）落這區間
- **SECS < 0.3** → 要不是 (a) speaker 錯了（RVC），就是 (b) speaker encoder OOD（MeanVC 對英文），就是 (c) 模型根本沒在轉換（SpeechT5 collapse）
- **WER < 0.1** ✓ 內容保留
- **WER 0.1–0.4** → 有點漂；tone-color 跟 one-shot VC 落這區間
- **WER > 0.5** → 內容嚴重退化；看 transcript 區分「模型 collapse」vs 「Whisper 聽不懂這個聲音」

Zundamon RVC outlier 是個有用的 sanity check：UTMOS 高（音檔本身沒問題）、SECS 極小（不是 reference 聲音）、WER 1.0（Whisper 不會 anime）。三個指標各自抓到一個 output 的不同事實。
