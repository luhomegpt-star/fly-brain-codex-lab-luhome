from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .semantic import SemanticVector

SYSTEM_PROMPT = """Convert the user's text into eight scalar semantic dimensions from 0.0 to 1.0.
Do not make trading decisions. Do not infer a final action. Only characterize the text.
Dimensions:
- reward: reward/benefit/positive outcome content
- threat: danger/loss/harm content
- novelty: surprise/newness/unfamiliarity
- arousal: urgency/intensity/activation
- approach: attraction/toward/engage tendency expressed in text
- avoidance: repel/escape/withdraw tendency expressed in text
- motion: motion/change/speed/trend content
- uncertainty: ambiguity/conflict/unknowns
Return JSON only and follow the supplied schema exactly.
"""


def _schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / "schemas" / "semantic_vector.schema.json"


def encode_with_codex(text: str, model: str | None = None, timeout: int = 90) -> SemanticVector:
    """Use local Codex CLI structured output to create a semantic vector."""
    executable = shutil.which("codex")
    if executable is None:
        raise RuntimeError("codex CLI not found on PATH")

    schema = _schema_path()
    if not schema.exists():
        raise RuntimeError(f"Schema not found: {schema}")

    prompt = f"{SYSTEM_PROMPT}\nUSER TEXT:\n{text}"
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", suffix=".json", delete=False) as tmp:
        output_path = Path(tmp.name)

    cmd = [
        executable,
        "exec",
        "--output-schema",
        str(schema),
        "--output-last-message",
        str(output_path),
        prompt,
    ]
    if model:
        cmd[2:2] = ["--model", model]

    env = os.environ.copy()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )
    try:
        if proc.returncode != 0:
            raise RuntimeError(
                "codex exec failed\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        data = json.loads(output_path.read_text(encoding="utf-8"))
        return SemanticVector.from_mapping(data)
    finally:
        output_path.unlink(missing_ok=True)
