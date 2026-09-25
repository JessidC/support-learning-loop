"""ABCD conversations (ASAPP, MIT) turned into what a support bot sees.

Each dialogue gives the customer's question, the agent's turns, and the system actions the agent
took. Informational cases (FAQ answers, site troubleshooting) can be solved from a knowledge base.
Cases where the agent had to verify identity, refund, or change an account need a person, and the
loop must never learn from them.
"""
from __future__ import annotations

import gzip
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

INFORMATIONAL_FLOWS = {"storewide_query", "single_item_query", "troubleshoot_site"}
# Action turns that only mean "the agent looked something up". Anything else touched an account.
LOOKUP_ACTIONS = re.compile(
    r"^(Searching the FAQ pages|System Action: search|FAQ answer related to|Agent is looking for solutions"
    r"|The website team has been notified|Details of .* have been entered)", re.I)
FAQ_SELECTED = re.compile(r"FAQ answer related to .* was selected", re.I)
GREETING = re.compile(r"^(hi|hello|hey|good (morning|afternoon|evening))\b[\s!.,]*$", re.I)


@dataclass
class Conversation:
    id: str
    split: str
    flow: str
    topic: str  # ABCD subflow; used only for analysis, never shown to the bot or the loop
    question: str
    agent_turns: list[str]
    actions: list[str]
    faq_answer: str = ""  # agent turns right after an FAQ answer was selected, if any
    turns: list[tuple[str, str]] = field(default_factory=list)

    @property
    def needs_person(self) -> bool:
        """True when the agent had to act on an account, not just look something up."""
        return any(not LOOKUP_ACTIONS.match(a) for a in self.actions)

    @property
    def informational(self) -> bool:
        return self.flow in INFORMATIONAL_FLOWS and not self.needs_person


def first_question(turns: list[tuple[str, str]]) -> str:
    """What the customer asked before the agent went to look it up.

    Customers often open with "I have a question about these boots" and say the actual question a
    turn later, so this joins the customer's messages up to the agent's first lookup, capped at
    70 words.
    """
    said = []
    for s, t in turns:
        if s == "action":
            break
        if s == "customer" and not GREETING.match(t.strip()):
            said.append(t.strip())
        if len(said) == 3:
            break
    words = " ".join(said).split()
    return " ".join(words[:70])


CLOSING = re.compile(r"anything else|have a (great|nice|good)|you're welcome|glad (i|to)", re.I)


def faq_answer(turns: list[tuple[str, str]]) -> str:
    """The agent's explanation right after an FAQ answer was selected (up to 3 turns, 90 words)."""
    out, armed = [], False
    for s, t in turns:
        if s == "action" and FAQ_SELECTED.search(t):
            armed = True
            continue
        if not armed:
            continue
        if s == "agent":
            if CLOSING.search(t):
                break
            out.append(t.strip())
            if len(out) == 3 or len(" ".join(out).split()) >= 90:
                break
    return " ".join(out)


def load(path: Path) -> list[Conversation]:
    raw = json.load(gzip.open(path, "rt", encoding="utf-8"))
    convs = []
    for split, items in raw.items():
        for c in items:
            turns = [(s, t) for s, t in c["original"]]
            convs.append(Conversation(
                id=str(c["convo_id"]), split=split, flow=c["scenario"]["flow"], topic=c["scenario"]["subflow"],
                question=first_question(turns),
                agent_turns=[t.strip() for s, t in turns if s == "agent"],
                actions=[t.strip() for s, t in turns if s == "action"],
                faq_answer=faq_answer(turns), turns=turns))
    return [c for c in convs if c.question]


def sample(convs: list[Conversation], split: str, n_info: int, n_person: int, seed: int) -> list[Conversation]:
    """A traffic mix: informational cases plus cases that need a person."""
    rng = random.Random(seed)
    pool = [c for c in convs if c.split == split]
    info = [c for c in pool if c.informational]
    person = [c for c in pool if c.needs_person]
    out = rng.sample(info, min(n_info, len(info))) + rng.sample(person, min(n_person, len(person)))
    rng.shuffle(out)
    return out
