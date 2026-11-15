"""Configuration of the answer-generation step (``src/generate/answer.py``).

The retrieval pipeline is frozen as ``bge-base``, top 5 chunks. The refusal gate refuses when the
top-1 cosine similarity is below ``gate_threshold``. Its value comes from the bge-base row of
``results/retrieval/refusal.json``, the threshold that keeps false refusals on the answerable eval
items at or below 5 % (0.690515). That value is exactly the top-1 score of one answerable item
(cmb-006), so a 1e-7 difference between GPU and CPU encoding decides whether that item is gated;
it is therefore rounded down to 0.6905, which makes the same decisions on every eval item (no
score lies in [0.6905, 0.690515) other than cmb-006's). It was chosen on the eval set, so it is
optimistic for new questions.

``pipeline`` selects the generation pipeline:

* ``"v1"``: retrieve, gate, LLM; the prompt asks for inline ``"quote" [Sn]`` in the answer; every
  refusal is :data:`REFUSAL_SENTENCE`. Kept so the v1 results can be reproduced.
* ``"v2"`` (default): the router (``src/generate/router.py``) runs before the gate (personal-advice
  redirect, drug grounding, weight-based dose lookup); the prompt asks for ``[Sn]`` labels after each
  claim and quotes only in ``citations``; every refusal shown to the user is :data:`REFUSAL_MESSAGE`
  (the LLM itself still replies with :data:`REFUSAL_SENTENCE`, which is then replaced).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

REFUSAL_SENTENCE = "This isn't covered in the guidelines provided."
REFUSAL_MESSAGE = "This isn't covered in the guidelines provided. Please consult a health professional."
REDIRECT_MESSAGE = ("I can't advise on your own or a family member's treatment. "
                    "Please speak to a doctor, nurse or pharmacist.")
PIPELINES = ("v1", "v2")


@dataclass(frozen=True)
class GenerationConfig:
    pipeline: str = "v2"
    retriever: str = "bge-base"
    top_k: int = 5
    gate_threshold: float = 0.6905  # results/retrieval/refusal.json bge-base (0.690515), rounded down
    retriever_device: str = "cpu"  # keeps the GPU for the LLM; cached chunk embeddings are reused
    model: str = "mistral"
    ollama_url: str = "http://localhost:11434"
    temperature: float = 0.0
    seed: int = 42
    num_ctx: int = 8192  # 5 chunks of up to ~800 tokens plus the prompt; Ollama's default is smaller
    num_gpu: int | None = None  # layers offloaded to the GPU; None lets Ollama decide (0 = CPU only)
    # max output tokens: stops a runaway generation (qwen2.5:7b looped past 6000 tokens on hd-008);
    # well above the longest mistral answer (555 tokens in v1 and v2, run before this cap existed)
    num_predict: int | None = 1024
    max_retries: int = 1  # one retry on invalid output
    request_timeout_s: float = 600.0

    def with_(self, **changes: Any) -> GenerationConfig:
        return replace(self, **changes)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
