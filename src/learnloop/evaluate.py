"""Measure the bot on held-out chats, before and after the loop's entries are added.

Informational chats should be resolved: answered, and correct against what the human agent said.
Chats that needed a person should still be handed over; a loop that teaches the bot to answer
those has made it worse, whatever the resolution rate says.
"""
from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .bot import respond
from .data import Conversation
from .kb import KB
from .llm import LLM

GRADER = """You grade a support bot's reply against what a human agent said to the same customer.
The agent's replies are the reference. Judge only whether the bot's reply gives the customer the
same correct information or steps for their question.
- "correct": matches the reference on what matters to the customer.
- "partial": right direction but missing or vague on something that matters.
- "wrong": contradicts the reference or answers a different question.
Reply with JSON only: {"grade": "correct" | "partial" | "wrong", "why": "<one sentence>"}"""


def grade(judge: LLM, c: Conversation, answer: str) -> str:
    ref = "\n".join(f"AGENT: {t}" for t in c.agent_turns)
    for _ in range(2):  # one retry on a malformed reply
        try:
            g = judge.complete_json(GRADER, f"Customer question: {c.question}\n\nReference:\n{ref}\n\nBot reply: {answer}",
                                    max_tokens=2000)
            return g.get("grade") if g.get("grade") in ("correct", "partial", "wrong") else "wrong"
        except Exception:
            continue
    return "error"


def run(convs: list[Conversation], kb: KB, bot: LLM, judge: LLM, out: Path, workers: int = 8) -> dict:
    def one(c: Conversation) -> dict:
        r = respond(bot, kb, c.question)
        row = {"conv": c.id, "informational": c.informational, "topic": c.topic, "question": c.question,
               "handoff": r.handoff, "reason": r.reason, "top_similarity": round(r.top_similarity, 3),
               "answer": r.answer, "grade": None}
        if not r.handoff:
            row["grade"] = grade(judge, c, r.answer)
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, convs))
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    s = summarize(rows)
    s["kb_entries"] = len(kb.entries)
    s["cost_usd"] = round(bot.cost() + judge.cost(), 4)
    (out / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    return s


def summarize(rows: list[dict]) -> dict:
    info = [r for r in rows if r["informational"]]
    person = [r for r in rows if not r["informational"]]
    answered = [r for r in info if not r["handoff"]]
    return {
        "informational": len(info),
        "needs_person": len(person),
        "resolved": _pct(sum(r["grade"] == "correct" for r in info), len(info)),
        "escalated": _pct(sum(r["handoff"] for r in info), len(info)),
        "wrong_shown": _pct(sum(r["grade"] in ("partial", "wrong") for r in answered), len(answered)),
        "needs_person_handed_over": _pct(sum(r["handoff"] for r in person), len(person)),
        "grader_errors": sum(r["grade"] == "error" for r in rows),
    }


def _pct(a: int, b: int) -> float:
    return round(100 * a / b, 1) if b else 0.0


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def compare(before: Path, after: Path) -> dict:
    load = lambda p: {r["conv"]: r for r in map(json.loads, (p / "results.jsonl").read_text(encoding="utf-8").splitlines())}
    a, b = load(before), load(after)
    ids = [i for i in a if i in b and a[i]["informational"]]
    ok = lambda r: r["grade"] == "correct"
    fixed = sum(not ok(a[i]) and ok(b[i]) for i in ids)
    broken = sum(ok(a[i]) and not ok(b[i]) for i in ids)
    sa, sb = summarize(list(a.values())), summarize(list(b.values()))
    return {"before": sa, "after": sb, "fixed": fixed, "broken": broken, "p_value": round(mcnemar_exact(fixed, broken), 5)}
