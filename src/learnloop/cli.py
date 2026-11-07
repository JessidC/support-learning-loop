"""loop: prepare the data, run the bot and the improvement loop, review, and measure before/after."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import data, evaluate, loop, review
from .bot import respond
from .kb import KB, Entry
from .llm import LLM

STATE = Path("state")
RUNS = Path("runs")


def _convs() -> dict[str, data.Conversation]:
    cfg = json.loads((STATE / "config.json").read_text())
    return {c.id: c for c in data.load(Path(cfg["abcd"]))}


def _ids(name: str) -> list[str]:
    return (STATE / name).read_text().split()


def cmd_prepare(a) -> None:
    convs = data.load(Path(a.abcd))
    train = [c for c in convs if c.split == "train" and c.informational and c.faq_answer]
    faq_topics = sorted({c.topic for c in train if c.flow in ("storewide_query", "single_item_query")})
    seed_topics = faq_topics[::2]  # the starting help center covers half of the FAQ topics
    rng = random.Random(a.seed)
    seed = []
    for t in seed_topics:
        c = rng.choice([c for c in train if c.topic == t])
        seed.append(Entry(id=f"s{c.id}", question=c.question, answer=c.faq_answer, source=c.id, approved_by="seed"))
    used = {e.source for e in seed}
    pool = [c for c in convs if c.id not in used]
    traffic = data.sample(pool, "train", a.traffic_info, a.traffic_person, seed=a.seed)
    test = data.sample(pool, "test", a.test_info, a.test_person, seed=a.seed + 1)
    STATE.mkdir(exist_ok=True)
    KB(seed).save(STATE / "kb_v0.jsonl")
    (STATE / "traffic.txt").write_text("\n".join(c.id for c in traffic))
    (STATE / "test.txt").write_text("\n".join(c.id for c in test))
    (STATE / "config.json").write_text(json.dumps({"abcd": str(Path(a.abcd).resolve()), "seed_topics": seed_topics,
                                                   "unseeded_topics": faq_topics[1::2]}, indent=2))
    print(f"seed KB: {len(seed)} entries covering {len(seed_topics)} of {len(faq_topics)} FAQ topics "
          f"(troubleshooting topics start uncovered)")
    print(f"traffic: {len(traffic)} chats, test: {len(test)} chats")


def cmd_learn(a) -> None:
    convs, kb = _convs(), KB.load(STATE / "kb_v0.jsonl")
    traffic = [convs[i] for i in _ids("traffic.txt")]
    bot, judge = LLM(a.bot_provider), LLM(a.judge_provider)
    with ThreadPoolExecutor(max_workers=8) as pool:
        replies = list(pool.map(lambda c: respond(bot, kb, c.question), traffic))
    escalated = [c for c, r in zip(traffic, replies) if r.handoff]
    queue = loop.load_queue(STATE / "queue.jsonl")
    seen_path = STATE / "seen.txt"
    seen = set(seen_path.read_text().split()) if seen_path.exists() else set()
    queue, stats, log = loop.run(escalated, kb, judge, queue, seen)
    loop.save_queue(STATE / "queue.jsonl", queue)
    seen_path.write_text("\n".join(sorted(seen)))
    (STATE / "learn_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
    summary = {"traffic": len(traffic), "handed_over": len(escalated), "loop": dict(stats),
               "pending": sum(q.status == "pending" for q in queue),
               "cost_usd": {"bot": round(bot.cost(), 4), "judge": round(judge.cost(), 4)}}
    (STATE / "learn_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def cmd_review(a) -> None:
    queue = loop.load_queue(STATE / "queue.jsonl")
    if a.judge_only:
        n = review.judge_only(queue)
        print(f"approved {n} candidates as judge-only (no person read them)")
    elif a.audit:
        n = review.audit(queue, a.audit)
        print(f"labeled {n} items")
    else:
        n = review.interactive(queue, a.reviewer)
        print(f"decided {n} candidates")
    loop.save_queue(STATE / "queue.jsonl", queue)


def cmd_recheck(a) -> None:
    queue = loop.load_queue(STATE / "queue.jsonl")
    hit = loop.recheck(queue)
    loop.save_queue(STATE / "queue.jsonl", queue)
    print(f"{len(hit)} candidates moved to rejected by the account check: {', '.join(hit) or '-'}")


def cmd_publish(a) -> None:
    kb0 = KB.load(STATE / "kb_v0.jsonl")
    seed = len(kb0.entries)
    new = review.to_entries(loop.load_queue(STATE / "queue.jsonl"))
    out = STATE / a.out
    kb0.add(new).save(out)
    who = Counter(e.approved_by for e in new)
    print(f"{out.name}: {seed} seed + {len(new)} learned entries ({dict(who)})")


def cmd_eval(a) -> None:
    convs, kb = _convs(), KB.load(Path(a.kb))
    test = [convs[i] for i in _ids("test.txt")]
    s = evaluate.run(test, kb, LLM(a.bot_provider), LLM(a.judge_provider), RUNS / a.name)
    print(json.dumps(s, indent=2))


def cmd_compare(a) -> None:
    print(json.dumps(evaluate.compare(Path(a.before), Path(a.after)), indent=2))


def cmd_agreement(a) -> None:
    print(json.dumps(review.agreement(loop.load_queue(STATE / "queue.jsonl")), indent=2))


def main() -> None:
    p = argparse.ArgumentParser(prog="loop")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare", help="build the seed KB and the traffic and test samples")
    s.add_argument("--abcd", required=True, help="path to abcd_v1.1.json.gz")
    s.add_argument("--traffic-info", type=int, default=500)
    s.add_argument("--traffic-person", type=int, default=150)
    s.add_argument("--test-info", type=int, default=250)
    s.add_argument("--test-person", type=int, default=80)
    s.add_argument("--seed", type=int, default=11)
    s.set_defaults(fn=cmd_prepare)

    for name, fn, help_ in (("learn", cmd_learn, "run the bot on traffic and the loop on its handoffs"),
                            ("eval", cmd_eval, "measure the bot on the held-out test chats")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--bot-provider", default="deepseek")
        s.add_argument("--judge-provider", default="orca")
        if name == "eval":
            s.add_argument("--kb", required=True)
            s.add_argument("--name", required=True)
        s.set_defaults(fn=fn)

    s = sub.add_parser("review", help="the expert step")
    s.add_argument("--reviewer", default="expert")
    s.add_argument("--judge-only", action="store_true", help="approve every pending candidate without a person")
    s.add_argument("--audit", type=int, default=0, help="blind-label N items to measure the judge")
    s.set_defaults(fn=cmd_review)

    sub.add_parser("recheck", help="apply the account check to candidates no person has decided on").set_defaults(fn=cmd_recheck)
    s = sub.add_parser("publish", help="build a KB from the seed and the approved candidates")
    s.add_argument("--out", default="kb_v1.jsonl", help="file name under state/")
    s.set_defaults(fn=cmd_publish)
    s = sub.add_parser("compare", help="paired before/after comparison")
    s.add_argument("before")
    s.add_argument("after")
    s.set_defaults(fn=cmd_compare)
    sub.add_parser("agreement", help="judge vs human agreement on audited items").set_defaults(fn=cmd_agreement)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
