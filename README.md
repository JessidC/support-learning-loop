# support-learning-loop
[![tests](https://github.com/JessidC/support-learning-loop/actions/workflows/tests.yml/badge.svg)](https://github.com/JessidC/support-learning-loop/actions/workflows/tests.yml)

A support bot that learns from the agents who take over its chats. When the bot hands a conversation to a person, the agent's reply is the best evidence of what the bot was missing. This loop turns those replies into new knowledge base entries, and an expert approves, edits or rejects every one before the bot can use it.

I run the same kind of loop in production on an ERP support chat, where it is the main reason the engine keeps improving between releases. This repo rebuilds it in the open on public data, so the design and the numbers can be checked.

## Results

Measured on 330 held-out chats: 250 the bot could solve with the right knowledge, and 80 that needed a person. Same bot and same model in every run; only the knowledge base changes. No person read the learned entries in these runs: the judge approved them (`judge-only`), and the last column adds the automatic account check described below.

| | Starting KB, 24 entries | After the loop, 122 entries | After the loop + account check, 120 entries |
|---|---|---|---|
| Resolved: answered and correct | 19.2% | 61.2% | **58.8%** |
| Handed to a person | 67.6% | 20.8% | 23.6% |
| Partial or wrong, among answers shown | 40.7% | 22.7% | 23.0% |
| Chats that needed a person and were handed over | 97.5% | 91.2% | 91.2% |

Paired on the same chats, starting KB against the checked KB: 103 fixed, 4 broken, exact McNemar p < 0.0001. The 2.4-point gap between the last two columns is not significant (4 fixed, 10 broken, p = 0.18).

What the loop did with the 488 chats the bot handed over during learning:

- **142 dropped by rule** because the agent had to act on an account. No model call.
- **293 chats became 98 candidates.** Demand counts the chats behind each one. The top two: items not showing in the cart (39 chats) and a slow site (33).
- **44 rejected by the judge** as not reusable, and **10 more by the grounding check** because the answer had a number the agent never said.
- 4 already covered by the KB.

Cost: about US$0.75 for learning and US$0.32 for both evaluations.

### The safety net

The first judge-only run showed what an unreviewed loop gets wrong. Two learned refund-policy entries listed what a refund requires ("order ID, username and email"), and with them in the KB the bot asked a customer for those details to look up a return it cannot see. That is the kind of entry an expert rejects.

So the grounding step now has an account check. An entry whose answer asks the customer for account details (order ID, username, email address, password, card number) is rejected by rule, with no model call. `loop recheck` applies it to an existing queue without touching decisions a person already made. On this queue it caught exactly those two entries, and in the re-run the bot no longer offers to look up an account.

What the check does not cover, and a reviewer still would:

- **7 of the 80 chats that needed a person get an answer, against 2 with the starting KB.** None of them asks for or promises account access. They are out-of-stock complaints and expired promo codes, where the store's agent took an action (filed the complaint, issued a new code) and the bot gives the right general policy without handing over. Catching these is a rule for the bot, not for the KB: hand over complaints and requests for an exception.
- **Near-duplicates.** The 98 candidates include five about rejected credit cards and four about search. The merge threshold is conservative on purpose: merging two different questions is worse than showing a reviewer two similar ones.
- **The starting point is weak by design.** Each seed entry comes from one chat, so the bot sometimes answers with a close but different FAQ. That is why partial or wrong answers start at 40.7%.

### Limits

- **The judge's agreement with people is not measured yet.** `loop review --audit 40` shows a person a blind sample of what the judge accepted and rejected, and `loop agreement` reports Cohen's kappa. The number goes here once a person has labeled the sample.
- **The grader and the loop's judge are the same model family** (Gemini). The bot is DeepSeek, so nothing grades its own answers, but a grader from a third family would be stronger.
- **ABCD is role-played.** The agents followed written guidelines, so their answers are cleaner and more consistent than a real support floor. A real loop sees more noise per chat and needs the expert step more, not less.

## How the loop works

For every chat the bot handed over:

1. **Skip what was already judged.** A fingerprint of the agent's turns means a re-run never pays the judge twice for the same content.
2. **Drop cases that needed a person.** If the agent had to verify identity, refund or change an account, nothing reusable can come out of it. These are dropped by rule, before any model call.
3. **A judge from another model family reads the agent's turns.** The bot answers with DeepSeek and the judge is Gemini, so the model that answers never grades its own work. The judge decides whether the agent said something other customers could reuse, and if so writes a generic question and answer, quoting the agent sentences that support it.
4. **Check the judge against the chat.** Every quote must appear in the agent's turns and every number in the answer must be one the agent said. An answer that asks the customer for account details fails too, since the bot cannot look up an account. Any entry that fails is rejected.
5. **Compare with what the bot already knows.** If the knowledge base already covers the question it is skipped. If a pending candidate asks the same thing, its demand goes up by one. Otherwise it becomes a new candidate.
6. **An expert decides.** `loop review` walks the queue by demand: approve, edit or reject. Only approved and edited entries reach the knowledge base, and each one records who approved it.

The bot itself has two gates. Retrieval first: if no entry is close enough to the question it hands over without calling the model. Then the model answers only from the retrieved entries, and hands over when they do not cover the question or the customer needs someone to look at their account.

## Data

The [Action-Based Conversations Dataset](https://github.com/asappresearch/abcd) (ABCD, ASAPP, MIT license): 10,042 chats between customers and agents of an online clothing store, written by people following the store's guidelines. Each chat records the system actions the agent took, which is how the loop tells an informational question (an FAQ answer, fixing a problem with the site) from a case that needs a person (refunds, account changes, identity checks).

The starting knowledge base covers half of the 48 FAQ topics, one entry each, taken from the agent's explanation in one chat per topic. Site troubleshooting starts uncovered. The loop then sees 650 training chats, and the bot is measured on 330 held-out test chats before and after.

## Run it

```bash
uv sync
curl -LO https://raw.githubusercontent.com/asappresearch/abcd/master/data/abcd_v1.1.json.gz

uv run loop prepare --abcd abcd_v1.1.json.gz
uv run loop learn                    # bot on the training chats, loop on its handoffs
uv run loop review --reviewer you    # approve, edit or reject each candidate
uv run loop recheck                  # re-apply the rule checks to undecided candidates (no model calls)
uv run loop publish                  # seed KB + approved entries, into state/kb_v1.jsonl (--out to rename)
uv run loop eval --kb state/kb_v0.jsonl --name before
uv run loop eval --kb state/kb_v1.jsonl --name after
uv run loop compare runs/before runs/after
```

Set `DEEPSEEK_API_KEY` for the bot and `ORCAROUTER_API_KEY` for the Gemini judge, or pass `--judge-provider anthropic` to use Claude Haiku with `ANTHROPIC_API_KEY`. Embeddings run locally with fastembed.

`loop review --judge-only` approves every pending candidate without a person, for unattended benchmark runs. Those entries are marked `judge-only` in the knowledge base. `loop review --audit 40` shows a person a blind sample of what the judge accepted and rejected, and `loop agreement` reports how often the judge agrees with them.

## Layout

```
src/learnloop/
  data.py      ABCD chats: question, agent turns, and whether the case needed a person
  kb.py        knowledge base and local embedding index
  bot.py       answer from the KB or hand over
  loop.py      fingerprint, rule filter, cross-family judge, grounding check, dedup by demand
  review.py    the expert step: review, audit, agreement
  evaluate.py  before/after measurement with a grader and an exact McNemar test
tests/         unit tests, no API calls
runs/          the results behind the table above (before, after, after-account-check)
results/       the starting KB, the learned KBs (kb_v1 judge-only, kb_v2 with the account check) and the queue the judge produced
```

## License

Code is MIT. ABCD is MIT, Copyright (c) 2021 ASAPP Research.
