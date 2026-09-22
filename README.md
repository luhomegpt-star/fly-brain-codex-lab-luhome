# fly-brain-codex-lab-luhome

實驗目標：建立「文字 → Codex 語意向量 → 果蠅 connectome → readout → 文字」的可檢驗管線。

目前第一版包含：
- 8 維 semantic vector JSON schema
- keyword baseline
- deterministic random control
- Codex CLI structured-output encoder
- proxy neural backend（只測 plumbing，不宣稱是生物模擬）
- real flybrain adapter scaffold
- `--compare` 與 `--debug`

## 安裝

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 快速測試

```bash
text-fly-bridge "市場突然暴跌，大家非常恐慌" --encoder keyword --backend proxy --debug
text-fly-bridge "市場突然暴跌，大家非常恐慌" --compare --backend proxy --no-codex --debug
pytest
```

如果本機已安裝並登入 Codex CLI：

```bash
text-fly-bridge "市場突然暴跌，大家非常恐慌" --encoder codex --backend proxy --debug
```

## 科學限制

目前 `proxy` backend 只驗證資料流，**不是果蠅神經模擬**。真正 connectome backend 目前會明確失敗，不會偷偷 fallback 到 proxy；這是刻意設計，避免產生看似科學、實際是假的神經活動結果。

下一步要在本機鎖定並驗證 `flybrain` API，才把 8 維語意向量映射到真實 sensory-neuron stimulus，接入真正 connectome simulation。
