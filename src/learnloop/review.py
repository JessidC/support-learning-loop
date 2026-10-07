"""The expert step. Nothing the loop proposes reaches the KB without a decision recorded here.

Three ways to use it:
  - interactive review of the pending queue: approve, edit or reject each candidate;
  - an audit: a person labels a random sample that mixes what the judge accepted and what it
    rejected, which is how the judge's agreement with people is measured;
  - judge-only approval, for unattended benchmark runs. Entries approved this way are marked
    "judge-only" in the KB, so it is always visible which knowledge no person has read.
"""
from __future__ import annotations

import random

from .kb import Entry
from .loop import Candidate


def show(c: Candidate) -> None:
    print("\n" + "-" * 72)
    print(f"{c.id}  demand {c.demand}  sources {', '.join(c.sources[:5])}")
    print(f"Q: {c.question}")
    if c.answer:
        print(f"A: {c.answer}")
    for e in c.evidence:
        print(f'   agent said: "{e}"')
    if c.status == "judge_rejected":
        print(f"(judge rejected: {c.judge_reason})")


def interactive(queue: list[Candidate], reviewer: str, ask=input) -> int:
    done = 0
    for c in sorted((q for q in queue if q.status == "pending"), key=lambda q: -q.demand):
        show(c)
        while True:
            a = ask("[a]pprove  [e]dit  [r]eject  [s]kip  [q]uit > ").strip().lower()
            if a in ("a", "e", "r", "s", "q"):
                break
        if a == "q":
            break
        if a == "s":
            continue
        if a == "e":
            new = ask("New answer: ").strip()
            if new:
                c.answer = new
            c.status = "edited"
        else:
            c.status = "approved" if a == "a" else "rejected"
        c.reviewer = reviewer
        done += 1
    return done


def audit(queue: list[Candidate], n: int, seed: int = 7, ask=input) -> int:
    """Blind labeling: the person sees the chat outcome, not the judge's verdict."""
    rng = random.Random(seed)
    accepted = [q for q in queue if q.status != "judge_rejected" and not q.human_label]
    rejected = [q for q in queue if q.status == "judge_rejected" and not q.human_label]
    pick = rng.sample(accepted, min(n // 2, len(accepted))) + rng.sample(rejected, min(n - n // 2, len(rejected)))
    rng.shuffle(pick)
    done = 0
    for c in pick:
        print("\n" + "-" * 72)
        print(f"Customer question: {c.question}")
        if c.answer:
            print(f"Proposed answer:  {c.answer}")
        a = ask("Would this help other customers with the same question? [y]es [n]o [q]uit > ").strip().lower()
        if a == "q":
            break
        if a in ("y", "n"):
            c.human_label = "reusable" if a == "y" else "not_reusable"
            done += 1
    return done


def judge_only(queue: list[Candidate]) -> int:
    n = 0
    for c in queue:
        if c.status == "pending":
            c.status, c.reviewer = "approved", "judge-only"
            n += 1
    return n


def to_entries(queue: list[Candidate]) -> list[Entry]:
    return [Entry(id=c.id, question=c.question, answer=c.answer, source=",".join(c.sources), approved_by=c.reviewer)
            for c in queue if c.status in ("approved", "edited")]


def agreement(queue: list[Candidate]) -> dict:
    """Cohen's kappa between the judge's verdict and the human label on audited items."""
    pairs = [(c.status != "judge_rejected", c.human_label == "reusable") for c in queue if c.human_label]
    n = len(pairs)
    if not n:
        return {"n": 0}
    agree = sum(j == h for j, h in pairs) / n
    pj = sum(j for j, _ in pairs) / n
    ph = sum(h for _, h in pairs) / n
    chance = pj * ph + (1 - pj) * (1 - ph)
    kappa = (agree - chance) / (1 - chance) if chance < 1 else 1.0
    return {"n": n, "agreement": round(agree, 3), "kappa": round(kappa, 3),
            "judge_yes_human_no": sum(j and not h for j, h in pairs),
            "judge_no_human_yes": sum(h and not j for j, h in pairs)}
