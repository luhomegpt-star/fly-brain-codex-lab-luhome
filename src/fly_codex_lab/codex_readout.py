from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path


READOUT_PROMPT = """You are translating an experiment result into Traditional Chinese.

Rules:
1. Describe only information present in the supplied JSON.
2. Do not claim the fruit fly understands language.
3. Clearly distinguish the semantic encoder from the biological connectome.
4. If backend is "proxy", explicitly say it is not a biological simulation.
5. If backend is "flybrain", describe neural/readout activity without turning it into a trading decision.
6. Do not give buy/sell advice.
7. Keep the answer concise: 3-6 bullet points.
"""


def explain_with_codex(data: dict, timeout: int = 90) -> str:
    executable = shutil.which("codex")
    if executable is None:
        raise RuntimeError("codex CLI not found on PATH")

    payload = json.dumps(data, ensure_ascii=False, indent=2)
    prompt = f"{READOUT_PROMPT}\nEXPERIMENT JSON:\n{payload}"

    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", suffix=".txt", delete=False) as tmp:
        output_path = Path(tmp.name)

    proc = subprocess.run(
        [
            executable,
            "exec",
            "--output-last-message",
            str(output_path),
            prompt,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )

    try:
        if proc.returncode != 0:
            raise RuntimeError(
                "codex readout failed\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        return output_path.read_text(encoding="utf-8").strip()
    finally:
        output_path.unlink(missing_ok=True)
