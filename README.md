# fly-brain-codex-lab-luhome

實驗目標：建立 **文字 → Codex 語意向量 → 果蠅 connectome → descending-neuron readout → Codex 繁中解讀** 的可檢驗管線。

目前包含：
- 8 維 semantic vector JSON schema
- keyword baseline
- deterministic random control
- Codex CLI structured-output encoder
- proxy neural backend（只測 plumbing，不宣稱是生物模擬）
- real `flybrain 0.1.x` MaleCNS backend
- descending-neuron trace
- DNp01 / DNa02 / DNg100 / MDN command-neuron activity
- optional Codex final readout explainer
- `--compare` / `--debug`

## 1. 基礎安裝

```bash
python -m venv .venv
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -e ".[dev]"
```

先驗證 proxy：

```bash
text-fly-bridge "市場突然暴跌，大家非常恐慌" --compare --backend proxy --no-codex --debug
pytest
```

## 2. Codex semantic encoder

本機需已安裝並登入 Codex CLI：

```bash
text-fly-bridge "市場突然暴跌，大家非常恐慌" --encoder codex --backend proxy --debug
```

這一步只做：

```text
自然語言
  ↓
Codex
  ↓
8 維 semantic vector
```

Codex 不直接控制果蠅行為。

## 3. 安裝真正 flybrain

```bash
pip install -e ".[brain,dev]"
python -m flybrain download
```

第一次下載/初始化完成後，可以跑真正 connectome：

```bash
text-fly-bridge "市場突然暴跌，大家非常恐慌" \
  --encoder codex \
  --backend flybrain \
  --steps 64 \
  --debug
```

64 steps 在預設 dt=20 ms 下約等於 1.28 秒模擬時間。

### 目前 semantic → sensory mapping

這是**我們設計的實驗 encoder**，不是果蠅會理解中文：

- `threat / avoidance / arousal / motion` → LC4 + LPLC2 looming/escape channel
- `novelty / motion / uncertainty / arousal` → LPLC1 approaching-object channel
- `approach / reward / motion / novelty` → LC10a chase-target channel

真正生物部分從這些 neuron injections 之後才開始：訊號進入固定 MaleCNS connectome，再讀 descending neurons。

## 4. 再讓 Codex 把神經結果翻成繁體中文

```bash
text-fly-bridge "市場突然暴跌，大家非常恐慌" \
  --encoder codex \
  --backend flybrain \
  --steps 64 \
  --explain \
  --debug
```

完整鏈：

```text
中文文字
  ↓
Codex semantic encoder
  ↓
8 維 semantic vector
  ↓
heuristic sensory mapper
  ↓
LC4 / LPLC2 / LPLC1 / LC10a
  ↓
166,700-neuron MaleCNS connectome
  ↓
descending-neuron trace
  ↓
DNp01 / DNa02 / DNg100 / MDN activity
  ↓
Codex（只翻譯結果）
  ↓
繁體中文摘要
```

## 科學限制

1. `proxy` backend 只驗證資料流，不是生物神經模擬。
2. `flybrain` backend 的 connectome wiring 是真實 MaleCNS-derived network，但 semantic-to-sensory encoder 是我們設計的 heuristic。
3. 果蠅沒有理解文字；Codex 先把文字轉成可注入的標量。
4. `top_firing_neurons` 在 real backend 目前依 descending-neuron decaying trace integral 排序；它不是精確 per-neuron spike count。
5. 此實驗目前不做自動交易與下單。
