"""The improvement loop: turns the agents' answers to escalated chats into candidate KB entries.

For every chat the bot handed over:
  1. Skip it if this exact agent content was already judged (content fingerprint, so a re-run
     never pays the judge twice).
  2. Drop it before any model call if the agent had to act on an account: that is a case for a
     person, and nothing reusable can come out of it.
  3. A judge from a different model family reads the agent's turns and decides whether they hold
     an answer other customers could reuse. If so it writes a generic question and answer, quoting
     the agent turns that support it.
  4. Check the quotes and the numbers against the agent's own words. Anything the judge added that
     the agent never said sends the item to the rejected pile, and so does an answer that asks the
     customer for account details: the bot cannot look up an account, so that entry teaches it to
     promise something only a person can do.
  5. Compare with what the KB already knows: already covered is skipped, a repeat adds demand to
     the pending candidate, anything else becomes a new candidate.
Nothing reaches the KB from here. An expert approves, edits or rejects every candidate (review.py).
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .data import Conversation
from .kb import KB, embed
from .llm import LLM

COVERED = 0.85  # question vs KB entry (question + answer) similarity: the KB already answers it
SAME = 0.88  # similarity above which two candidates are the same question

JUDGE = """You review a customer support chat that an AI bot handed to a human agent. Decide whether
the agent's replies contain knowledge the bot could reuse to answer OTHER customers asking the same
thing, without a person.

Not reusable: the answer depends on this customer's own account, order, payment or identity; the
agent only asked for details, apologized, or said to contact someone or open a ticket; the agent
never actually answered.

If reusable, write a knowledge base entry:
- "question": the customer's need as a generic question, no names, order numbers or personal details.
- "answer": 1-3 sentences using ONLY facts the agent stated. Do not add steps, numbers or policies
  the agent did not say.
- "evidence": the exact agent sentences (copied verbatim) that support the answer.

Reply with JSON only:
{"reusable": true, "question": "...", "answer": "...", "evidence": ["...", "..."]}
or
{"reusable": false, "reason": "<short reason>"}"""

NUMBER = re.compile(r"\$?\d+(?:[.,]\d+)?%?")
# Details only someone with access to the customer's account can use. An entry that tells the
# customer to provide them describes a case for a person, not an answer the bot can give.
ACCOUNT = re.compile(r"\b(order (?:id|number)s?|user ?names?|e-?mail address(?:es)?|account (?:id|number)s?|"
                     r"passwords?|card numbers?|zip ?codes?|shipping address(?:es)?)\b", re.I)


def asks_for_account(answer: str) -> bool:
    return bool(ACCOUNT.search(answer))


@dataclass
class Candidate:
    id: str
    question: str
    answer: str
    evidence: list[str]
    sources: list[str]
    demand: int = 1
    status: str = "pending"  # pending | approved | edited | rejected | judge_rejected
    judge_reason: str = ""
    reviewer: str = ""
    human_label: str = ""  # "reusable" / "not_reusable" when a person audited this item
    topics: list[str] = field(default_factory=list)  # analysis only


def fingerprint(c: Conversation) -> str:
    return hashlib.sha256("\n".join(c.agent_turns).encode("utf-8")).hexdigest()[:16]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9$%]+", " ", s.lower()).strip()


def grounded(item: dict, c: Conversation) -> str | None:
    """Return a reason if the judge's entry says something the agent did not."""
    agent = _norm(" ".join(c.agent_turns))
    quotes = [q for q in item.get("evidence", []) if q.strip()]
    if not quotes:
        return "no_evidence"
    if not all(_norm(q) and _norm(q) in agent for q in quotes):
        return "evidence_not_in_chat"
    said = set(NUMBER.findall(" ".join(c.agent_turns)))
    if any(n not in said for n in NUMBER.findall(item.get("answer", ""))):
        return "number_not_in_chat"
    if asks_for_account(item.get("answer", "")):
        return "needs_account"
    return None


def recheck(queue: list[Candidate]) -> list[str]:
    """Apply the account check to candidates no person has decided on (pending or judge-only).

    Runs no model: it lets an existing queue pick up a check added after it was built.
    """
    hit = []
    for c in queue:
        undecided = c.status == "pending" or (c.status == "approved" and c.reviewer == "judge-only")
        if undecided and asks_for_account(c.answer):
            c.status, c.judge_reason, c.reviewer = "judge_rejected", "needs_account", ""
            hit.append(c.id)
    return hit


def judge_one(judge: LLM, c: Conversation) -> dict:
    chat = "\n".join(f"{s.upper()}: {t}" for s, t in c.turns if s in ("customer", "agent"))
    err = "unknown"
    for _ in range(2):  # one retry: a malformed reply is a transport problem, not a verdict
        try:
            return judge.complete_json(JUDGE, f"Chat:\n{chat}", max_tokens=2500)
        except Exception as exc:
            err = exc.__class__.__name__
    return {"error": err}


def run(escalated: list[Conversation], kb: KB, judge: LLM, queue: list[Candidate], seen: set[str],
        workers: int = 8) -> tuple[list[Candidate], Counter, list[dict]]:
    stats, log = Counter(), []
    todo = []
    for c in escalated:
        fp = fingerprint(c)
        if fp in seen:
            stats["already_judged"] += 1
            continue
        seen.add(fp)
        if c.needs_person:
            stats["dropped_needs_person"] += 1
            log.append({"conv": c.id, "outcome": "dropped_needs_person"})
            continue
        if sum(len(t.split()) for t in c.agent_turns) < 15:
            stats["dropped_no_content"] += 1
            log.append({"conv": c.id, "outcome": "dropped_no_content"})
            continue
        todo.append(c)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        verdicts = list(pool.map(lambda c: judge_one(judge, c), todo))

    pending = [q for q in queue if q.status == "pending"]
    pending_vecs = embed([q.question for q in pending])
    for c, v in zip(todo, verdicts):
        if "error" in v:  # not a verdict: forget the fingerprint so the next run tries again
            stats["judge_error"] += 1
            seen.discard(fingerprint(c))
            log.append({"conv": c.id, "outcome": "judge_error", "reason": v["error"]})
            continue
        if not v.get("reusable"):
            stats["judge_rejected"] += 1
            queue.append(Candidate(id=f"r{c.id}", question=c.question, answer="", evidence=[], sources=[c.id],
                                   status="judge_rejected", judge_reason=str(v.get("reason", "")), topics=[c.topic]))
            log.append({"conv": c.id, "outcome": "judge_rejected", "reason": v.get("reason")})
            continue
        why = grounded(v, c)
        if why:
            stats[f"ungrounded_{why}"] += 1
            queue.append(Candidate(id=f"r{c.id}", question=v.get("question", c.question), answer=v.get("answer", ""),
                                   evidence=v.get("evidence", []), sources=[c.id], status="judge_rejected",
                                   judge_reason=why, topics=[c.topic]))
            log.append({"conv": c.id, "outcome": "ungrounded", "reason": why})
            continue
        qv = embed([v["question"]])[0]
        hits = kb.search(v["question"], k=1)
        if hits and hits[0][1] >= COVERED:
            stats["already_covered"] += 1
            log.append({"conv": c.id, "outcome": "already_covered", "kb": hits[0][0].id})
            continue
        if len(pending):
            sims = pending_vecs @ qv
            j = int(np.argmax(sims))
            if sims[j] >= SAME:
                p = pending[j]
                p.demand += 1
                p.sources.append(c.id)
                if c.topic not in p.topics:
                    p.topics.append(c.topic)
                stats["added_demand"] += 1
                log.append({"conv": c.id, "outcome": "added_demand", "candidate": p.id})
                continue
        cand = Candidate(id=f"c{c.id}", question=v["question"], answer=v["answer"], evidence=v["evidence"],
                         sources=[c.id], topics=[c.topic])
        queue.append(cand)
        pending.append(cand)
        pending_vecs = np.vstack([pending_vecs, qv[None, :]]) if len(pending_vecs) else qv[None, :]
        stats["new_candidate"] += 1
        log.append({"conv": c.id, "outcome": "new_candidate", "candidate": cand.id})
    return queue, stats, log


def load_queue(path: Path) -> list[Candidate]:
    if not path.exists():
        return []
    return [Candidate(**json.loads(l)) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def save_queue(path: Path, queue: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(q), ensure_ascii=False) + "\n" for q in queue), encoding="utf-8")
