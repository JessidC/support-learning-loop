import numpy as np
import pytest

from learnloop import data, evaluate, loop, review
from learnloop.loop import Candidate


def conv(turns, flow="storewide_query", cid="1"):
    return data.Conversation(id=cid, split="train", flow=flow, topic="t", question=data.first_question(turns),
                             agent_turns=[t for s, t in turns if s == "agent"],
                             actions=[t for s, t in turns if s == "action"], faq_answer=data.faq_answer(turns),
                             turns=turns)


FAQ_CHAT = [
    ("agent", "Hi! How can I help?"),
    ("customer", "Hello"),
    ("customer", "I have a question about these boots"),
    ("customer", "how do I get gum off the sole?"),
    ("action", "Searching the FAQ pages ..."),
    ("action", "FAQ answer related to boots (other2) was selected."),
    ("agent", "Rub peanut butter over the gum and wait 10 minutes."),
    ("agent", "Then scrape it off and clean the sole."),
    ("customer", "thanks"),
    ("agent", "Anything else I can help with?"),
]


def test_question_joins_messages_until_the_lookup_and_skips_greetings():
    assert data.first_question(FAQ_CHAT) == "I have a question about these boots how do I get gum off the sole?"


def test_faq_answer_takes_the_explanation_and_stops_at_the_closing():
    assert data.faq_answer(FAQ_CHAT) == "Rub peanut butter over the gum and wait 10 minutes. Then scrape it off and clean the sole."


def test_account_actions_mean_the_case_needs_a_person():
    assert not conv(FAQ_CHAT).needs_person
    assert conv(FAQ_CHAT).informational
    refund = FAQ_CHAT + [("action", "A refund has been made for the amount of $40.")]
    assert conv(refund).needs_person
    assert not conv(refund).informational


def test_grounding_rejects_quotes_and_numbers_the_agent_never_said():
    c = conv(FAQ_CHAT)
    ok = {"answer": "Rub peanut butter on the gum, wait 10 minutes and scrape it off.",
          "evidence": ["Rub peanut butter over the gum and wait 10 minutes."]}
    assert loop.grounded(ok, c) is None
    assert loop.grounded({**ok, "evidence": ["Use ice to freeze the gum."]}, c) == "evidence_not_in_chat"
    assert loop.grounded({**ok, "answer": "Wait 30 minutes."}, c) == "number_not_in_chat"
    assert loop.grounded({**ok, "evidence": []}, c) == "no_evidence"


class FakeKB:
    def __init__(self, score):
        self.score = score

    def search(self, q, k=1):
        return [(type("E", (), {"id": "kb1"})(), self.score)]


class FakeJudge:
    def __init__(self, verdict):
        self.verdict = verdict

    def complete_json(self, system, user, max_tokens=0):
        return self.verdict


GOOD = {"reusable": True, "question": "How do I remove gum from boots?",
        "answer": "Rub peanut butter over the gum and scrape it off.",
        "evidence": ["Rub peanut butter over the gum and wait 10 minutes."]}


@pytest.fixture(autouse=True)
def fake_embed(monkeypatch):
    monkeypatch.setattr(loop, "embed", lambda texts: np.ones((len(texts), 4), dtype=np.float32) / 2)


def test_loop_drops_account_cases_before_calling_the_judge():
    refund = conv(FAQ_CHAT + [("action", "Account has been pulled up for Ana Ruiz.")])

    class Boom:
        def complete_json(self, *a, **k):
            raise AssertionError("judge must not be called")

    queue, stats, _ = loop.run([refund], FakeKB(0.1), Boom(), [], set())
    assert stats["dropped_needs_person"] == 1 and not queue


def test_loop_creates_then_adds_demand_and_never_judges_the_same_content_twice():
    seen = set()
    a, b = conv(FAQ_CHAT, cid="1"), conv(FAQ_CHAT + [("agent", "Glad I could help with the boots today.")], cid="2")
    queue, stats, _ = loop.run([a, b, a], FakeKB(0.1), FakeJudge(GOOD), [], seen)
    assert stats["new_candidate"] == 1 and stats["added_demand"] == 1 and stats["already_judged"] == 1
    assert queue[0].demand == 2 and queue[0].sources == ["1", "2"]


def test_loop_skips_what_the_kb_already_covers():
    queue, stats, _ = loop.run([conv(FAQ_CHAT)], FakeKB(0.95), FakeJudge(GOOD), [], set())
    assert stats["already_covered"] == 1 and not queue


def test_judge_rejections_are_kept_for_the_audit():
    queue, stats, _ = loop.run([conv(FAQ_CHAT)], FakeKB(0.1), FakeJudge({"reusable": False, "reason": "asked for data"}), [], set())
    assert stats["judge_rejected"] == 1 and queue[0].status == "judge_rejected"


def test_only_approved_or_edited_candidates_reach_the_kb():
    q = [Candidate("a", "q1", "a1", [], ["1"], status="approved", reviewer="jarby"),
         Candidate("b", "q2", "a2", [], ["2"], status="pending"),
         Candidate("c", "q3", "a3", [], ["3"], status="rejected"),
         Candidate("d", "q4", "a4", [], ["4"], status="edited", reviewer="jarby")]
    assert [e.id for e in review.to_entries(q)] == ["a", "d"]
    assert review.judge_only(q) == 1 and q[1].reviewer == "judge-only"


def test_interactive_review_records_decisions_and_edits():
    q = [Candidate("a", "q1", "a1", [], ["1"], demand=3), Candidate("b", "q2", "a2", [], ["2"], demand=1)]
    answers = iter(["e", "Better answer", "r"])
    assert review.interactive(q, "jarby", ask=lambda _: next(answers)) == 2
    assert q[0].status == "edited" and q[0].answer == "Better answer" and q[1].status == "rejected"


def test_agreement_kappa():
    q = [Candidate(str(i), "q", "a", [], [], status=s, human_label=h) for i, (s, h) in enumerate([
        ("pending", "reusable"), ("pending", "reusable"), ("judge_rejected", "not_reusable"),
        ("judge_rejected", "reusable"), ("pending", "not_reusable"), ("judge_rejected", "not_reusable")])]
    r = review.agreement(q)
    assert r["n"] == 6 and r["agreement"] == pytest.approx(4 / 6, abs=1e-3)
    assert r["judge_yes_human_no"] == 1 and r["judge_no_human_yes"] == 1


def test_summary_and_mcnemar():
    rows = [
        {"informational": True, "handoff": False, "grade": "correct"},
        {"informational": True, "handoff": False, "grade": "wrong"},
        {"informational": True, "handoff": True, "grade": None},
        {"informational": False, "handoff": True, "grade": None},
        {"informational": False, "handoff": False, "grade": "partial"},
    ]
    s = evaluate.summarize(rows)
    assert s["resolved"] == pytest.approx(33.3) and s["escalated"] == pytest.approx(33.3)
    assert s["wrong_shown"] == 50.0 and s["needs_person_handed_over"] == 50.0
    assert evaluate.mcnemar_exact(0, 0) == 1.0
    assert evaluate.mcnemar_exact(10, 0) < 0.01
