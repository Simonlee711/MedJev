"""Local MLX engine for MedJev and the zero-shot LLM baselines.

The frozen MedJev helper (MedJev-Flash-9B-MLX/helper/shim.py, sha 81a22f1b...) renders every question and turns letter
logits into Jev answers. We load that exact file, as shim_mlx.py does, and replace only its model client, so
prompts and calibration are identical to the HTTP server. Any MLX model can be plugged in with the same prompts and
readout (chat models through their own chat template; base models with plain=True), each read out from its own
tokenizer's option-letter tokens. Optionally the last-token hidden states of selected decoder layers are recorded in
the same forward pass, and the shared "State:" prefix of a multi-question request can be computed once and reused for
every question (prefix cache; the same computation as separate passes, up to float rounding).

Runs in the `mlx` environment (mlx, mlx-lm, transformers):  mlx/bin/python -m ...  or import from scripts run there.
"""

import copy
import hashlib
import importlib.util
import os
import types
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models import qwen3_5
from mlx_lm.models.cache import make_prompt_cache

ROOT = Path(__file__).resolve().parents[2]
MEDJEV = ROOT / "MedJev-Flash-9B-MLX"
HELPER = MEDJEV / "helper" / "shim.py"
HELPER_SHA_PREFIX = "81a22f1b"

# MedJev's published readout settings; see src/scripts/serve_medjev.sh.
READOUT_ENV = {"READOUT_T": "1.07", "READOUT_NOUL_T": "1.074766", "READOUT_NOUL_BIAS": "0",
               "READOUT_TARGETED": "1", "READOUT_INSTR_STYLE": "pyrepr", "SHIM_STAGGER": "1"}

CAPTURE_LAYERS = (8, 12, 16, 20, 24, 28)
# MLX keeps freed GPU buffers for reuse; with a different prompt length every call that cache grows without bound
# (tens of GB over a long run). Cap it, and clear it every CLEAR_EVERY requests.
CACHE_LIMIT_BYTES = 2 * 1024**3
CLEAR_EVERY = 25  # decoder layers whose last-token output is recorded; "final" = after norm

# ----------------------------------------------------------------------------------------------------
# Hidden-state capture: same computation as Qwen3_5TextModel.__call__, recording the last position.

CAPTURE: dict = {"on": False, "layers": CAPTURE_LAYERS, "out": None}


def _forward_with_capture(self, inputs, cache=None, input_embeddings=None):
    h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
    if cache is None:
        cache = [None] * len(self.layers)
    fa_mask = qwen3_5.create_attention_mask(h, cache[self.fa_idx])
    ssm_mask = qwen3_5.create_ssm_mask(h, cache[self.ssm_idx])
    keep = {}
    for i, (layer, c) in enumerate(zip(self.layers, cache)):
        h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=c)
        if CAPTURE["on"] and (i + 1) in CAPTURE["layers"]:
            keep[str(i + 1)] = h[:, -1]
    out = self.norm(h)
    if CAPTURE["on"]:
        keep["final"] = out[:, -1]
        CAPTURE["out"] = keep
    return out


qwen3_5.Qwen3_5TextModel.__call__ = _forward_with_capture


class _Obj(types.SimpleNamespace):
    pass


def load_helper():
    """Import the frozen helper with the MedJev readout settings (it reads them at import time)."""
    sha = hashlib.sha256(HELPER.read_bytes()).hexdigest()
    assert sha.startswith(HELPER_SHA_PREFIX), f"helper is {sha[:16]}, not the frozen {HELPER_SHA_PREFIX}"
    for k, v in READOUT_ENV.items():
        os.environ.setdefault(k, v)
    os.environ.setdefault("TOKENIZER", str(MEDJEV))
    os.environ.setdefault("SHIM_MODEL", str(MEDJEV))
    spec = importlib.util.spec_from_file_location("medjev_helper", HELPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.TARGETED and mod.PERMS == 1
    return mod


class Engine:
    def __init__(self, model_dir: str | Path = MEDJEV, adapter_path: str | Path | None = None,
                 capture: bool = True, prefix_cache: bool = False, plain: bool = False, max_tokens: int | None = None):
        """plain=True: a base model without a chat template (e.g. Pythia). The rendered prompt is followed by
        "\nAnswer:" and the readout uses the model's own " A", " B", ... tokens. max_tokens: context window; longer
        prompts keep their last max_tokens tokens (the end holds the question and options)."""
        mx.set_cache_limit(CACHE_LIMIT_BYTES)
        self.model, _ = load(str(model_dir), adapter_path=str(adapter_path) if adapter_path else None)
        self._calls = 0
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.helper = load_helper()
        self.helper.client = _Obj(chat=_Obj(completions=_Obj(create=self._create)), base_url="mlx://local")
        self.plain, self.max_tokens = plain, max_tokens
        # Letter tokens of *this* model's tokenizer (the helper would otherwise cache MedJev's). For chat models the
        # answer starts the assistant turn ("A"); plain models continue "Answer:" (" A"). For MedJev this is unchanged.
        prefix = " " if plain else ""
        self.helper._ids.clear()
        for L in self.helper.LETTERS:
            self.helper._ids[L] = self.tok.encode(prefix + L, add_special_tokens=False)[0]
        assert all(len(self.tok.encode(prefix + L, add_special_tokens=False)) == 1 for L in self.helper.LETTERS[:26]), \
            "option letters are not single tokens for this tokenizer"
        # Letter ids come from the MedJev tokenizer; the base Qwen3.5 tokenizer is the same vocabulary.
        self.capture = capture
        self.prefix_cache = prefix_cache
        self._prefix = {"ids": None, "cache": None}
        self._hidden = None
        self.prompts: list[str] = []  # rendered prompts of the last answer() call
        self.dry_run = False

    # ---- the helper's model client
    def _ids(self, content: str) -> list[int]:
        if self.plain:
            ids = self.tok.encode(content + "\nAnswer:", add_special_tokens=False)
        else:
            ids = self.tok.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                               add_generation_prompt=True, enable_thinking=False)
            ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        if self.max_tokens and len(ids) > self.max_tokens:
            ids = ids[-self.max_tokens:]
        return list(ids)

    def _last_logits(self, ids: list[int], content: str):
        cut = content.rfind("\n\nQuestion: ") if self.prefix_cache else -1
        if cut < 0:
            return self.model(mx.array(ids)[None])[0, -1]
        shared = self._prefix["ids"]
        if not (shared and ids[: len(shared)] == shared and len(ids) > len(shared)):
            head = self._ids(content[:cut])
            n = next((i for i, (x, y) in enumerate(zip(ids, head)) if x != y), min(len(ids), len(head)))
            n = min(n, len(ids) - 1)
            if n < 64:
                return self.model(mx.array(ids)[None])[0, -1]
            cache = make_prompt_cache(self.model)
            was = CAPTURE["on"]
            CAPTURE["on"] = False
            for start in range(0, n, 2048):
                self.model(mx.array(ids[start:min(n, start + 2048)])[None], cache=cache)
                mx.eval([c.state for c in cache])
            CAPTURE["on"] = was
            self._prefix = {"ids": ids[:n], "cache": cache}
            shared = ids[:n]
        return self.model(mx.array(ids[len(shared):])[None], cache=copy.deepcopy(self._prefix["cache"]))[0, -1]

    def _create(self, model=None, max_tokens=1, temperature=0, logprobs=False, messages=None, extra_body=None, **kw):
        eb = extra_body or {}
        want = eb.get("logprob_token_ids") or eb.get("allowed_token_ids")
        content = messages[0]["content"]
        self.prompts.append(content)
        ids = self._ids(content)
        if self.dry_run:
            vals, self._hidden = [0.0] * len(want), None
        else:
            CAPTURE["on"] = self.capture
            logits = self._last_logits(ids, content).astype(mx.float32)
            lp = logits - mx.logsumexp(logits, axis=-1)
            vals = np.array(lp[mx.array(want)]).tolist()
            if self.capture:
                hid = CAPTURE["out"]
                mx.eval(list(hid.values()))
                self._hidden = {k: np.array(v[0].astype(mx.float16)) for k, v in hid.items()}
            CAPTURE["on"] = False
        top = [_Obj(token=f"token_id:{i}", logprob=v) for i, v in zip(want, vals)]
        return _Obj(choices=[_Obj(logprobs=_Obj(content=[_Obj(top_logprobs=top)]))], usage=_Obj(prompt_tokens=len(ids)))

    # ---- public API
    def answer(self, state, questions: dict) -> tuple[dict, dict]:
        """Jev answers for one request, and per question the recorded hidden states {layer: vector}."""
        st = self.helper.with_image(state)
        self._prefix = {"ids": None, "cache": None}
        self.prompts = []
        answers, hidden = {}, {}
        for qid, q in questions.items():
            ans, _ = self.helper.ANSWER[q["type"]](st, q)
            answers[qid] = ans
            hidden[qid] = self._hidden
        self._prefix = {"ids": None, "cache": None}
        self._calls += 1
        if self._calls % CLEAR_EVERY == 0:
            mx.clear_cache()
        return answers, hidden

    def render(self, state, questions: dict) -> dict[str, str]:
        """The exact prompt text of every question (no forward pass)."""
        self.dry_run = True
        try:
            st = self.helper.with_image(state)
            out = {}
            for qid, q in questions.items():
                self.prompts = []
                self.helper.ANSWER[q["type"]](st, q)
                out[qid] = self.prompts[-1]
            return out
        finally:
            self.dry_run = False

    def letter_ids(self, n: int) -> list[int]:
        ids = self.helper.letter_ids()
        return [ids[self.helper.LETTERS[i]] for i in range(n)]
