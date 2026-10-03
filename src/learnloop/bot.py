"""The support bot: answers from the knowledge base or hands the chat to a person.

Two gates decide a handoff. Retrieval first: if no entry is close enough to the question, the bot
does not try. Then the model: it must answer only from the retrieved entries and say so when they
do not cover the question.
"""
from __future__ import annotations

from dataclasses import dataclass

from .kb import KB
from .llm import LLM

MIN_SIMILARITY = 0.72

SYSTEM = """You answer customer support chats for an online clothing store, using ONLY the numbered
knowledge base entries you are given. If the entries do not answer the customer's question, or the
question needs someone to look at their account, order or payment, hand the chat to a person.
Reply with JSON only:
{"answer": "<reply to the customer, 1-3 sentences>", "handoff": false, "used": [<entry numbers>]}
or
{"answer": "", "handoff": true, "reason": "<short reason>"}"""


@dataclass
class Reply:
    handoff: bool
    answer: str
    reason: str
    top_similarity: float


def respond(llm: LLM, kb: KB, question: str) -> Reply:
    hits = kb.search(question, k=3)
    top = hits[0][1] if hits else 0.0
    if top < MIN_SIMILARITY:
        return Reply(True, "", "no_match", top)
    entries = "\n".join(f"[{i + 1}] Q: {e.question}\n    A: {e.answer}" for i, (e, _) in enumerate(hits))
    try:
        out = llm.complete_json(SYSTEM, f"Knowledge base:\n{entries}\n\nCustomer: {question}", max_tokens=400)
    except Exception as exc:  # a failed call is a handoff, never a made-up answer
        return Reply(True, "", f"error: {exc.__class__.__name__}", top)
    if out.get("handoff") or not str(out.get("answer", "")).strip():
        return Reply(True, "", out.get("reason") or "model_handoff", top)
    return Reply(False, str(out["answer"]).strip(), "", top)
