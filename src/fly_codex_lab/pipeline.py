from __future__ import annotations
from .codex_bridge import encode_with_codex
from .fly_backend import get_backend
from .semantic import keyword_encode, random_encode

def encode_text(text, encoder, seed=5175, model=None):
    if encoder == 'keyword': return keyword_encode(text)
    if encoder == 'random': return random_encode(text, seed)
    if encoder == 'codex': return encode_with_codex(text, model=model)
    raise ValueError(encoder)

def run_pipeline(text, encoder='keyword', backend='proxy', seed=5175, steps=64, model=None):
    vec = encode_text(text, encoder, seed, model)
    fly = get_backend(backend, seed).run(vec, steps).to_dict()
    return {'text': text, 'encoder': encoder, 'semantic_vector': vec.to_dict(), 'fly_result': fly}

def compare_encoders(text, backend='proxy', seed=5175, steps=64, include_codex=True, model=None):
    encoders = ['keyword', 'random'] + (['codex'] if include_codex else [])
    results = {}
    for encoder in encoders:
        try:
            results[encoder] = run_pipeline(text, encoder, backend, seed, steps, model)
        except Exception as exc:
            results[encoder] = {'error': f'{type(exc).__name__}: {exc}'}
    return {'text': text, 'backend': backend, 'results': results}
