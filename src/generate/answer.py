"""Answer a question from the guidelines: retrieve, route, gate, generate with citations, validate.

Pipeline (``Answerer.answer``; ``config.pipeline`` selects v1 or v2):

1. Retrieve the top 5 chunks with ``bge-base`` (frozen retrieval pipeline).
2. **Router** (v2 only, ``src/generate/router.py``), in order:

   a. personal advice -> the fixed redirect (:data:`~src.generate.config.REDIRECT_MESSAGE`);
   b. a drug named in the question appears in none of the top-5 chunks -> refuse;
   c. weight in kg + an infection in ``aware_dosing.csv`` -> the matching weight-band rows and their
      chunks go to the LLM (``route: "dose_lookup"``); no matching row -> refuse.

   None of these calls the LLM except c.
3. **Refusal gate**: if the top-1 cosine similarity is below ``config.gate_threshold``, refuse
   without calling the LLM.
4. Otherwise call the local LLM (Ollama, ``config.model``) with the five chunks labelled
   ``[S1]``..``[S5]``, each with its document title, section path and printed page(s). For each
   source the prompt gives:

   * ``Text``: the chunk's ``source_text`` plus its footnotes, the only text that may be quoted
     (quotes are verified against exactly this, ``src/generate/verify.py``);
   * ``Reading guide`` (flowcharts only): the derived ``flow_text`` (box order and branches,
     with bracketed editorial notes on printed errors), marked as not quotable;
   * ``Editorial note`` (where set): the chunk's reviewed ``source_discrepancy``, marked as not
     quotable.

   ``context_header`` and ``aware_category_note`` are not shown (the title / section line
   replaces the header; the AWaRe category is not needed to answer).

   v1 asks for an inline verbatim ``"quote" [Sn]`` in every factual sentence; v2 asks for an
   ``[Sn]`` label after each claim, with the quotes only in ``citations``.
5. The model must return JSON ``{"answer", "citations": [{"label", "quote"}], "refused"}``
   (Ollama structured output with :data:`OUTPUT_SCHEMA`). The output is validated
   (:func:`parse_output`); on invalid output the model is told what was wrong and asked once
   more. If it is still invalid, the result has ``valid: false`` and no answer.

Refusals: v1 shows :data:`~src.generate.config.REFUSAL_SENTENCE` for every refusal; v2 shows
:data:`~src.generate.config.REFUSAL_MESSAGE` (the LLM still replies with the short sentence, which is
then replaced). ``result["route"]`` is one of ``redirect``, ``drug_refusal``, ``dose_lookup``,
``gate`` or ``llm``, and ``result["refusal_source"]`` says what refused (``redirect``,
``drug_grounding``, ``dose_no_match``, ``gate``, ``llm``) or is None.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from src.evaluate.relevance import load_chunks
from src.generate.config import REDIRECT_MESSAGE, REFUSAL_MESSAGE, REFUSAL_SENTENCE, GenerationConfig
from src.generate.llm import ChatModel, OllamaClient, OllamaError
from src.generate.router import (
    DoseLookup,
    DrugLexicon,
    build_lexicon,
    detect_personal_advice,
    dose_row_text,
    lookup_doses,
    ungrounded_drugs,
)
from src.parse.schema import split_text, verifiable_text
from src.parse.text_utils import normalise_for_match
from src.retrieve.base import Retriever

SYSTEM_PROMPT_V1 = f"""You answer questions about WHO clinical guidelines using ONLY the numbered sources in the user message.

Rules:
1. Use only information stated in the sources. Do not use outside knowledge.
2. Every factual sentence in "answer" must contain at least one verbatim quote from a source's Text, in double quotes, immediately followed by the source label in square brackets, for example: Adults should receive amoxicillin "500 mg given every 8 hours" [S2].
3. Quotes must be copied character for character from the "Text" of a source. Never quote a "Reading guide" or an "Editorial note": they explain the source but are not quotable.
4. List every quote you used in "citations" as {{"label": "S2", "quote": "<the exact quoted words>"}}.
5. If the sources do not answer the question, set "answer" to exactly: {REFUSAL_SENTENCE}
   and set "refused" to true and "citations" to []. Otherwise "refused" is false.
6. If the sources conflict, present both versions, each with its own quote, and say that they conflict.
7. If the question asks for advice about the asker's own treatment (for example "should I take / stop / change my ..."), do not give advice: say that they should consult a health professional about their own treatment.
8. Never state a dose, frequency or duration that is not written in the sources.

Reply with only a JSON object of this form:
{{"answer": "<answer text>", "citations": [{{"label": "S1", "quote": "<verbatim quote>"}}], "refused": false}}"""

SYSTEM_PROMPT_V2 = f"""You answer questions about WHO clinical guidelines using ONLY the numbered sources in the user message.

Rules:
1. Use only information stated in the sources. Do not use outside knowledge.
2. In "answer", put the label of the supporting source in square brackets after each claim, for example: Adults should receive amoxicillin 500 mg every 8 hours for 5 days [S2]. Every sentence that contains a number must carry a label. Do not put quotations in "answer".
3. Put the supporting quotes only in "citations": for every label used in "answer", give at least one quote copied character for character from the "Text" of that same source, as {{"label": "S2", "quote": "<the exact words from the Text of S2>"}}. Never quote a "Reading guide", a "Dose table row" or an "Editorial note": they explain the source but are not quotable.
4. If the sources do not answer the question, set "answer" to exactly: {REFUSAL_SENTENCE}
   and set "refused" to true and "citations" to []. Otherwise "refused" is false.
5. If the sources conflict, present both versions, each with its own label, and say that they conflict.
6. If the question asks for advice about the asker's own treatment (for example "should I take / stop / change my ..."), do not give advice: say that they should consult a health professional about their own treatment.
7. Never state a dose, frequency or duration that is not written in the sources.

Reply with only a JSON object of this form:
{{"answer": "<answer text with [S1] labels>", "citations": [{{"label": "S1", "quote": "<verbatim quote>"}}], "refused": false}}"""

SYSTEM_PROMPTS = {"v1": SYSTEM_PROMPT_V1, "v2": SYSTEM_PROMPT_V2}
SYSTEM_PROMPT = SYSTEM_PROMPT_V2

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"label": {"type": "string"}, "quote": {"type": "string"}},
                "required": ["label", "quote"],
            },
        },
        "refused": {"type": "boolean"},
    },
    "required": ["answer", "citations", "refused"],
}


@dataclass(frozen=True)
class Source:
    label: str  # "S1".."S5"
    chunk_id: str
    score: float | None  # retrieval score; None for dose-lookup sources
    record: dict[str, Any]


def page_label(record: dict[str, Any]) -> str:
    start, end = record.get("printed_page_start"), record.get("printed_page_end")
    if start is None:
        start, end = record["pdf_page_start"], record["pdf_page_end"]
        return f"PDF p. {start}" if start == end else f"PDF pp. {start}-{end}"
    return f"p. {start}" if start == end else f"pp. {start}-{end}"


def source_block(source: Source) -> str:
    rec = source.record
    lines = [f"[{source.label}] {rec['doc']} | {' > '.join(rec['section_path'])} | {page_label(rec)}",
             "Text:", verifiable_text(rec)]
    if rec.get("flow_text"):
        _, body, _, _ = split_text(rec)  # flow_text with typo corrections, as in the retrieval text
        lines += ["Reading guide (derived from the flowchart; not quotable):", body]
    if rec.get("source_discrepancy"):
        lines += ["Editorial note (reviewed source discrepancy; not quotable):", rec["source_discrepancy"]]
    return "\n".join(lines)


def build_user_prompt(question: str, sources: list[Source]) -> str:
    labels = ", ".join(s.label for s in sources)
    blocks = "\n\n".join(source_block(s) for s in sources)
    return (f"Question: {question}\n\nSources ({labels}):\n\n{blocks}\n\n"
            f"Answer the question following the rules. Use only the labels {labels}.")


def build_dose_prompt(question: str, lookup: DoseLookup, sources: list[Source]) -> str:
    """User prompt for the weight-based dose route: the matching dose-table rows, then their chunks."""
    label_of = {s.chunk_id: s.label for s in sources}
    labels = ", ".join(s.label for s in sources)
    rows = "\n".join(dose_row_text(r, label_of[r["chunk_id"]]) for r in lookup.rows)
    blocks = "\n\n".join(source_block(s) for s in sources)
    return (f"Question: {question}\n\n"
            f"Dose table rows for {lookup.weight_kg:g} kg, {lookup.infection} (looked up from the guideline's "
            f"dosing tables; each row is labelled with the source it comes from; not quotable):\n{rows}\n\n"
            f"Sources ({labels}):\n\n{blocks}\n\n"
            "Answer the question from the dose table rows: for each row give the drug, dose, frequency and "
            "duration for this weight, with the source label. If the rows list several drugs, they are options: "
            "list them all, and say which are first or second choice where the rows say so. "
            f"Quote the supporting words from the Text of each source in \"citations\". Use only the labels {labels}.")


def is_refusal(answer: str) -> bool:
    a = normalise_for_match(answer)
    return a in (normalise_for_match(REFUSAL_SENTENCE), normalise_for_match(REFUSAL_MESSAGE))


def normalise_label(label: str) -> str:
    return label.strip().strip("[]").strip().upper()


def parse_output(raw: str, labels: list[str]) -> tuple[dict[str, Any] | None, list[str]]:
    """Parse and validate the model's JSON. Returns ``(parsed, errors)``; ``parsed`` is None if invalid."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, [f"not valid JSON ({exc.msg})"]
    if not isinstance(obj, dict):
        return None, ["output is not a JSON object"]
    errors: list[str] = []
    answer, citations, refused = obj.get("answer"), obj.get("citations"), obj.get("refused")
    if not isinstance(answer, str) or not answer.strip():
        errors.append('"answer" must be a non-empty string')
    if not isinstance(refused, bool):
        errors.append('"refused" must be true or false')
    if not isinstance(citations, list):
        errors.append('"citations" must be a list')
        citations = []
    clean: list[dict[str, str]] = []
    for i, c in enumerate(citations):
        if not isinstance(c, dict) or not isinstance(c.get("label"), str) or not isinstance(c.get("quote"), str):
            errors.append(f"citations[{i}] must be {{\"label\": str, \"quote\": str}}")
            continue
        label = normalise_label(c["label"])
        if label not in labels:
            errors.append(f"citations[{i}] has unknown label {c['label']!r} (use {', '.join(labels)})")
        if not c["quote"].strip():
            errors.append(f"citations[{i}] has an empty quote")
        clean.append({"label": label, "quote": c["quote"]})
    if isinstance(answer, str) and isinstance(refused, bool):
        if refused and not is_refusal(answer):
            errors.append(f'"refused" is true, so "answer" must be exactly: {REFUSAL_SENTENCE}')
        if not refused and is_refusal(answer):
            errors.append('"answer" is the refusal sentence, so "refused" must be true')
    if errors:
        return None, errors
    return {"answer": answer.strip(), "citations": clean, "refused": refused}, []


class Answerer:
    def __init__(self, config: GenerationConfig = GenerationConfig(), retriever: Retriever | None = None,
                 llm: ChatModel | None = None, chunks: dict[str, dict[str, Any]] | None = None,
                 lexicon: DrugLexicon | None = None) -> None:
        self.config = config
        self.chunks = chunks if chunks is not None else {c["chunk_id"]: c for c in load_chunks()}
        self.retriever = retriever if retriever is not None else self._load_retriever()
        self.lexicon = lexicon if lexicon is not None else build_lexicon(list(self.chunks.values()))
        options: dict[str, Any] = {"temperature": config.temperature, "seed": config.seed, "num_ctx": config.num_ctx}
        if config.num_gpu is not None:
            options["num_gpu"] = config.num_gpu
        if config.num_predict is not None:
            options["num_predict"] = config.num_predict
        self.llm = llm if llm is not None else OllamaClient(config.model, config.ollama_url, options=options,
                                                            timeout_s=config.request_timeout_s)

    def _load_retriever(self) -> Retriever:
        from src.retrieve.corpus import load_corpus
        from src.retrieve.dense import DENSE_SPECS, DenseRetriever

        return DenseRetriever.load_or_build(DENSE_SPECS[self.config.retriever], load_corpus(),
                                            device=self.config.retriever_device)

    @property
    def refusal_text(self) -> str:
        return REFUSAL_SENTENCE if self.config.pipeline == "v1" else REFUSAL_MESSAGE

    def retrieve(self, question: str) -> list[Source]:
        hits = self.retriever.search(question, self.config.top_k)
        return [Source(f"S{i}", cid, score, self.chunks[cid]) for i, (cid, score) in enumerate(hits, start=1)]

    def answer(self, question: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        sources = self.retrieve(question)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        gate_score = sources[0].score if sources else float("-inf")
        retrieved = [{"label": s.label, "chunk_id": s.chunk_id, "score": round(s.score, 6)} for s in sources]
        result: dict[str, Any] = {
            "question": question,
            "pipeline": self.config.pipeline,
            "retrieved": retrieved,
            "sources": retrieved,  # what the LLM is shown (differs from retrieved on the dose route)
            "gate_score": round(gate_score, 6),
            "gate_threshold": self.config.gate_threshold,
            "gated": False,
            "route": None,
            "route_detail": None,
            "attempts": [],
        }

        def finish(**fields: Any) -> dict[str, Any]:
            result.update(fields)
            llm_ms = sum(a["llm_ms"] for a in result["attempts"])
            result["latency_ms"] = {"retrieval": round(retrieval_ms, 1), "llm": round(llm_ms, 1),
                                    "total": round((time.perf_counter() - t0) * 1000, 1)}
            return result

        def fixed(route: str, text: str, refusal_source: str, detail: Any = None) -> dict[str, Any]:
            return finish(route=route, route_detail=detail, valid=True, refused=True, refusal_source=refusal_source,
                          answer=text, citations=[], parsed=None)

        if self.config.pipeline == "v2":
            if detect_personal_advice(question, self.lexicon):
                return fixed("redirect", REDIRECT_MESSAGE, "redirect")
            missing = ungrounded_drugs(question, [s.record for s in sources], self.lexicon)
            if missing:
                return fixed("drug_refusal", REFUSAL_MESSAGE, "drug_grounding", {"drugs": missing})
            lookup = lookup_doses(question, self.lexicon)
            if lookup is not None:
                detail = {"weight_kg": lookup.weight_kg, "infection": lookup.infection, "drugs": list(lookup.drugs),
                          "rows": [{k: r[k] for k in ("antibiotic", "population", "weight_band", "line", "resolved_dose",
                                                      "resolved_frequency", "duration", "chunk_id")}
                                   for r in lookup.rows]}
                if not lookup.rows:
                    return fixed("dose_lookup", REFUSAL_MESSAGE, "dose_no_match", detail)
                dose_sources = [Source(f"S{i}", cid, None, self.chunks[cid])
                                for i, cid in enumerate(lookup.chunk_ids(), start=1)]
                result["sources"] = [{"label": s.label, "chunk_id": s.chunk_id, "score": None} for s in dose_sources]
                result.update(route="dose_lookup", route_detail=detail)
                return finish(**self._generate(build_dose_prompt(question, lookup, dose_sources), dose_sources, result))

        if gate_score < self.config.gate_threshold:
            result["gated"] = True
            return fixed("gate", self.refusal_text, "gate")
        result["route"] = "llm"
        return finish(**self._generate(build_user_prompt(question, sources), sources, result))

    def _generate(self, user_prompt: str, sources: list[Source], result: dict[str, Any]) -> dict[str, Any]:
        labels = [s.label for s in sources]
        messages = [{"role": "system", "content": SYSTEM_PROMPTS[self.config.pipeline]},
                    {"role": "user", "content": user_prompt}]
        parsed = None
        for _ in range(1 + self.config.max_retries):
            t1 = time.perf_counter()
            try:
                reply = self.llm.chat(messages, format=OUTPUT_SCHEMA)
            except (OllamaError, OSError) as exc:  # includes timeouts: counts as an invalid attempt
                ms = (time.perf_counter() - t1) * 1000
                result["attempts"].append({"raw": "", "errors": [f"LLM call failed: {exc!r}"], "llm_ms": round(ms, 1),
                                           "stats": {}})
                continue
            ms = (time.perf_counter() - t1) * 1000
            parsed, errors = parse_output(reply.content, labels)
            result["attempts"].append({"raw": reply.content, "errors": errors, "llm_ms": round(ms, 1),
                                       "stats": reply.stats})
            if parsed is not None:
                break
            messages += [{"role": "assistant", "content": reply.content},
                         {"role": "user", "content": "Your reply was invalid: " + "; ".join(errors)
                          + ". Reply again with only the corrected JSON object, following all the rules."}]
        refused = bool(parsed and parsed["refused"])
        answer = parsed["answer"] if parsed else None
        if refused:
            answer = self.refusal_text
        return {"parsed": parsed, "valid": parsed is not None, "refused": refused,
                "refusal_source": "llm" if refused else None, "answer": answer,
                "citations": parsed["citations"] if parsed else []}


class PrecomputedRetriever:
    """Serves hits computed earlier (``retrieve_hits``), so the embedding model need not stay in memory."""

    def __init__(self, name: str, hits: dict[str, list[tuple[str, float]]]) -> None:
        self.name = name
        self.hits = hits

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        return self.hits[query][:k]


def retrieve_hits(questions: list[str], config: GenerationConfig) -> dict[str, list[tuple[str, float]]]:
    """Top-k hits for each question with the configured retriever (run in a child process by the
    eval script, so the model's memory is returned to the OS before the LLM runs)."""
    from src.retrieve.corpus import load_corpus
    from src.retrieve.dense import DENSE_SPECS, DenseRetriever

    retriever = DenseRetriever.load_or_build(DENSE_SPECS[config.retriever], load_corpus(), device=config.retriever_device)
    return {q: [(cid, float(score)) for cid, score in retriever.search(q, config.top_k)] for q in questions}
