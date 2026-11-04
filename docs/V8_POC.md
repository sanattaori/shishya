# A one-hour POC: separate memories and a sharp read, in place of the big table (plan, before any result)

*Written 3 Nov 2026, 20:12 IST, while the dev data is being collected, before any read-design score was seen. The author
asked for what can be done in one hour on this CPU. This does not replace v8 or change any past verdict. Code:
`brain/poc_collect.py`, `brain/poc_read.py`.*

## The problem

v7 keeps all its memories mixed together in one table. Reading it adds up everything that looks a bit like the question.
A person's facts share units and their meanings are close (V7.md: cosine 0.98–0.99), so they blur. v7's write traces
already hold each memory apart, in far less room: 4.3 MB against 268 MB for its long life.

## The design

- The same frozen 100M BDH, the same W, the same address (who × what). **No new training and no gradients.**
- Each memory stays apart, as a trace. There is no table.
- A question finds its closest memories. Closeness counts as s ** power (sharp).
- Among memories within `near` of the best match, the **newest** answers. `newest = false` is a plain sharp sum.

## Dev choices (dev names, 16 people; the CPU can't do 40 in the time)

A grid of power, near, gain and newest. The best dev mean of (changes current value, never-changed, one sentence plain)
wins, and the first in grid order wins a tie. v7's table on the same questions is the control, and should land near
v7's dev numbers (47.5% and 73.3%).

## Rules (test names, seed 1, once; 20 people, fewer than v7's 40, so numbers are less exact)

The same bars as v7:

| Rule | Passes when |
|---|---|
| P1 changes, current value, plain + other words | ≥ 70% |
| P2 never-changed facts | ≥ 85% |
| P3 one sentence, plain / other words | ≥ 80% / ≥ 60% |

Also checked, as a win over v7 if it holds: a paired gain on the same questions, against v7's table in the same run.

**Not tested in this hour:** v7's Y4 (school, life test) and Y5 (the five-day long life), and real sentences unless time
is left. So a pass here is not a pass for v8.

## Predictions (before any score)

- P3 one sentence: rises a lot, likely above v7's 39%, 60–80% plain. This is the kind of blur a sharp read should fix.
- P2 never-changed: rises to 80–95%.
- P1 changes: the risk. "Moved to" is stored far from "lives in", so the old fact can still be the closest match, and
  newest-wins only helps when both are close. 40–65%, likely fails.
- Overall: fails on P1, with real gains on P2 and P3. About 25% to pass all three.

## What happens next (fixed now)

- **P2 and P3 rise a lot, P1 doesn't:** the read design is right and the remaining problem is meaning. Say which words
  fail, and try roles next on this design.
- **All pass:** run Y4 and Y5 on Kaggle before any claim.
- **Nothing rises:** the table was not the limit. Say so plainly, and don't build on this.

## Amendment, 3 Nov 2026, 20:37 IST (after the dev read check and control, before any test data or score)

- **Control:** v7's table on the 16 dev people gives changes 45%, never-changed 81%, one sentence 42.5% / 39%, close to
  v7's dev numbers. So the collected data is sound.
- **What the check showed on dev:** for changed facts, the right memory is ahead of the best wrong option on 3% of plain
  questions. The closest memory is usually the old wording. So the predicted risk on P1 is real, and sharp reading alone
  can't fix it.
- **Added: roles.** A 32 × 32 map on the "what" vector, trained once, with gradients, on the 26 childhood relations only
  (train names, none of the five tested relations). Wordings of one kind of fact are pulled together and other kinds
  pushed apart. Life stays gradient-free. `power`, `near`, `gain` and `newest` are chosen again on dev, by the same rule.
- **Test size:** 12 people, not 20, to fit the hour. This was chosen for time, before any test data. Numbers get wider,
  so only large gains count.
- **Prediction for roles:** P1 rises but stays under 70%; moving to a town has to be linked to living in a town across
  relations it never saw.

## Amendment 2, 3 Nov 2026, 20:53 IST (after the dev sweep, before any test score)

**The dev sweep:** no setting of the new read beats v7's table. The dev mean is 0.564 for the table and 0.528 for the best
new read: changes fall from 45% to 28%, never-changed from 81% to 79%, and one sentence plain rises from 42.5% to 51%. So
the prediction that separate memories fix the table was wrong on changes. The table's delta rule overwrites better than a
newest-wins rule on these addresses.

**Added on dev, same rule:** the table rebuilt from traces (its score must equal v7's), the table with the role map, and
the table plus a sharp read added together. The best dev mean over all of them, the plain table included, is the one test
candidate. The table itself is the control on the same test questions. If the table wins dev, there is no candidate.

## Amendment 3, 3 Nov 2026, 21:03 IST (after dev results, before any test score)

**Dev so far:** the rebuilt table equals v7's exactly (the control holds). The best dev mean is the table plus a sharp sum
(power 4, near 0.9, gain 4): 0.576 against 0.564, only about 1 point, which is noise on 16 people. It stays the one test
candidate from amendment 2. The role map hurts: the table falls to 35% on never-changed facts.

**Added: a Guru at the door.** Before the Brain stores a sentence, a teacher restates it as simple one-fact sentences in
the Brain's existing wording for that kind of fact. The Brain, its address and table are v7's. There is no LLM at question
time. Step 1 is a perfect restatement, which is only a ceiling. Step 2 is a real free LLM doing the restating. Only step 2
counts as a result.

**Predictions (before any score).** Perfect: changes 75–95%, one sentence plain 80–95%, never-changed about the same as
v7. Real free LLM: 5–15 points lower.

## Results (3 Nov 2026, 21:15 IST; test names, 12 people, once; run on this CPU)

The test candidate (amendment 2) is v7's table with a sharp sum added (power 4, near 0.9, gain 4). The control is v7's table
on the same questions. Intervals are 95% paired bootstrap, 10,000 resamples.

| Rule | Needs | v7's table | Table + sharp read | Gain | |
|---|---|---:|---:|---|---|
| P1 changes, current value (48) | ≥ 70% | 41.7% | 43.8% | +2.1 [0, +6.2] | fail |
| P2 never-changed facts (36) | ≥ 85% | 69.4% | **91.7%** | **+22.2 [+11.1, +36.1]** | **pass** |
| P3 one sentence, plain (60) | ≥ 80% | 35% | 41.7% | +6.7 [−5, +18.3] | fail |
| P3 one sentence, other words (60) | ≥ 60% | 28.3% | 33.3% | +5 [−5, +15] | fail |

**Verdict: FAIL overall (P1, P3); P2 passes with a clear gain.** On 16 dev people never-changed rose only from 81% to 88%,
so the size of the gain is not settled. This is 12 people, and Y4 and Y5 (older tests, the long life) were not run.

**What was tried and lost, dev names (mean of changes, never-changed, one sentence plain; v7's table 0.564):**
- Separate memories with no table: best 0.528. One sentence rose from 42.5% to 51%, but changes fell from 45% to 28%.
- The role map: the table with it falls to 0.387 (never-changed 35%). Its best with a sharp read is 0.562.

**The limit, measured on dev:** at the step that picks a changed fact's first letter, the right memory beats the best wrong
option on only 3% of plain questions. The old wording is usually closest. Sharp reading can't fix that.

**Guru, perfect restatement (a ceiling, dev, one sentence):** plain 62.5%, other words 51%, against the table's 42.5% and
39%. Below the predicted 80–95%, even with no restating errors. The real free LLM (step 2) was not run in the hour.

**Guru, perfect restatement, changes (a ceiling, dev, 16 people; added 21:26 IST):** current value 92.2% (plain 90.6%, other
words 93.8%), against the table's 45.3%. Never-changed 72.9%, against 81.3%. It was inside the predicted 75–95%. So the
delta rule does replace an old value when the new one arrives in the same wording, and the wording gap causes most of the
change failures. The restatements here use the exam's own plain wording, so this is the best case, not a result.

## Amendment 4: a real free AI as the Guru, 3 Nov 2026, about 23:00 IST (before any score with a real teacher)

The author agreed to a big AI helping the Brain learn, at least for now, and asked for this test. Only this step counts as a
result for the Guru. So far, only seven dev sentences were restated, to pick a working model and prompt. Nothing was scored.

**Model:** `nvidia/nemotron-3.5-super-vl:free`, temperature 0. No paid model.

**The teachers (the exam's wordings are not in either prompt, and neither sees the world's facts):**
- **Plain:** it gets one sentence at a time, and says each fact in ordinary present-tense words.
- **Notebook:** the same, and it also sees what it already stored about the names in the sentence. If a sentence gives a new
  value for a stored fact, it copies that stored sentence and swaps only the value. The notebook is extra text kept outside
  the Brain. It is used when learning only, and that is a real cost to report.

**Dev choice:** teacher × (table, or table + the sharp read already fixed in amendment 3: power 4, near 0.9, gain 4, sum), 16
dev people. The best dev mean of (changes, never-changed, one sentence plain) is the one test candidate.

**Test:** 12 fresh people, seed 2, once. Seed 1 was already used on the sharp read, so it is not used again. The control is
v7's table with no teacher, on the same people and questions. No retries or edits to what the teacher writes. A reply that
cannot be read as a list counts as the original sentence stored unchanged.

**Rules:** the same bars. P1 changes ≥ 70%. P2 never-changed ≥ 85%. P3 one sentence ≥ 80% plain, ≥ 60% other words. **The go/no-go,
told to the author before this run: P1 ≥ 70% on test, or the Guru idea has failed as tested.** Also reported, from the world's
facts and only for reading the result: how many of the teacher's sentences kept the right person and value.

**Predictions (before any score).** Amendment 3 said 5–15 points below perfect, and that stands on the record. It assumed the
exam's wording, so I now expect less: changes 55–80%, most likely about 65%, so P1 about 40% likely to pass. Never-changed
70–90%. One sentence 45–65% plain, 35–50% other words, so P3 very likely fails. The notebook teacher beats the plain one on
changes. All three rules passing: under 10%.

## Amendment 5, 4 Nov 2026, 00:20 IST (after dev, before any test score with a teacher)

**Dev, real free AI, 16 people (no teacher: changes 45.3%, never-changed 81.3%, one sentence 42.5% / 38.8%):**

| Teacher and read | Changes | Never-changed | One sentence, plain / other | Dev mean |
|---|---:|---:|---:|---:|
| Plain, table | 89.1% | 72.9% | 43.8% / 45.0% | 0.686 |
| Plain, table + sharp | 73.4% | 85.4% | 52.5% / 40.0% | 0.705 |
| Notebook, table | 85.9% | 79.2% | 42.5% / 43.8% | 0.692 |
| **Notebook, table + sharp** | **78.1%** | **95.8%** | **51.2% / 41.2%** | **0.751** |

**The one test candidate, by the rule in amendment 4:** the notebook teacher with table + sharp. The other sharp settings in
the same log were not eligible, and were not considered. The dev changes score is above my predicted 55–80% for the table
alone, and the one-sentence score is below the perfect teacher's 62.5%, as predicted.

**Added, because the dev score was higher than predicted: a check on the harness.** A teacher that returns each sentence
unchanged goes through the same code on the same test people. It must land near the no-teacher control (changes 45.8%). If it
doesn't, this run's gains come from the code path, not the teacher, and the result does not count.

**Also reported, for information only, not the candidate:** the notebook teacher with the table alone, on test.

## Results: a real free AI as the Guru (4 Nov 2026, 00:55 IST; 12 fresh people, seed 2, once; run on this CPU)

The candidate is the notebook teacher with table + sharp (amendment 5). The control is v7's table with no teacher, on the same
questions. Intervals are 95% paired bootstrap, 10,000 resamples.

| Rule | Needs | No teacher | Candidate | Gain | |
|---|---|---:|---:|---|---|
| P1 changes, current value (48) | ≥ 70% | 45.8% | 68.8% | +22.9 [0, +45.8] | **fail, by one question** |
| P2 never-changed facts (36) | ≥ 85% | 66.7% | **88.9%** | **+22.2 [+8.3, +38.9]** | **pass** |
| P3 one sentence, plain (60) | ≥ 80% | 36.7% | 50.0% | +13.3 [+1.7, +25] | fail |
| P3 one sentence, other words (60) | ≥ 60% | 30.0% | 38.3% | +8.3 [−1.7, +18.3] | fail |

**Verdict: FAIL (P1, P3). By the go/no-go fixed in amendment 4, the Guru idea failed as tested.** P1 needed 34 of 48 and got
33. That closeness does not change the verdict, and a pass by one question would not have been firm either.

**The harness check holds.** A teacher that changes nothing gives changes 45.8% and never-changed 66.7%, the same as the control,
and one sentence 38.3% / 33.3%, within two questions of it. So the gains come from the teacher, not the code path.

**For information, not the candidate:** the notebook teacher with the table alone gives changes 77.1% (+31.2 [+10.4, +52.1]),
never-changed 69.4%, one sentence 41.7% / 33.3%. It would pass P1 and fail P2, and this can't replace the candidate after the
fact. The sharp read trades one for the other on every comparison so far: on dev it lowers changes by 8–16 points, and on test
by 8. It raises never-changed by 13–17 on dev, and by 19 on test.

**The teacher's work:** no reply was unreadable. Person and value stayed in one sentence in 132 of 132 lessons and 60 of 60
one-sentence facts. This is a name check, not a check of meaning. It still writes one kind of fact in several ways (such as "lives
in" and "home is in"), and sometimes adds empty sentences ("Everyone knows Dobun."). Time: about 8 s a sentence on the free model.

**Predictions:** changes 68.8% was inside 55–80%, and my most likely value was 65%. Never-changed, 88.9%, was inside 70–90%. One
sentence, 50.0% / 38.3%, was inside 45–65% / 35–50%. All three passing (under 10%) did not happen. "The notebook teacher beats the
plain one on changes" was mixed on dev (lower with the table, higher with the sharp read) and was not tested on test.

**Not tested:** real text, combining facts, the older tests and the five-day life. Seed 2 is now used.

## Amendment 6: hide replaced values from the sharp read, 4 Nov 2026, about 01:30 IST (before any score of this fix)

**The guess being tested:** the sharp read still sees old values after a fact has changed, and they pull answers back. The author
agreed to try this, then the cause of the one-sentence failure, then real text, and to rethink if those show no clear progress.

**Step 1, a ceiling, dev only:** hide every old value from the sharp read using the exam's own knowledge of which are old. The
table is untouched. If changes don't rise well above 78.1% here, the guess is wrong, and I stop this fix and say so.

**Step 2, the real rule (no world knowledge):** when the teacher stores a sentence that differs from an earlier one only in its
last name, the earlier one's value is hidden. Other ways of telling replacements apart may be tried on dev, and I record them
here before any test score.

**Test:** 12 fresh people, seed 3, once. Control is v7's table with no teacher. Teacher: the notebook one, same model and prompt.

**Rules:** P1 changes ≥ 70%, P2 never-changed ≥ 85%, P3 one sentence ≥ 80% plain, ≥ 60% other words. **Go/no-go, told to the
author: P1 and P2 both pass; then the full exam and five-day life on Kaggle.** P3 is expected to fail, as this fix doesn't touch it.

**Predictions (before any score):** ceiling on dev: changes 80–92%. Real rule on test: changes 65–82%, never-changed 80–92%. P1 and
P2 both pass: about 50%. One sentence unchanged within 5 points of the last run.

## Amendment 7, 4 Nov 2026, about 01:40 IST (after dev, before any test score of this fix)

**Dev, notebook teacher, table + sharp, 16 people:**

| Hidden from the sharp read | Changes | Never-changed | One sentence, plain / other |
|---|---:|---:|---:|
| Nothing (the last test's design) | 78.1% | 95.8% | 51.2% / 41.2% |
| Ceiling: every old value, from the exam's own list | 85.9% | 95.8% | 51.2% / 41.2% |
| **Real rule: same words, only the last name differs** | **84.4%** | **95.8%** | **51.2% / 41.2%** |

The ceiling is inside the 80–92% I predicted, and the real rule gets nearly all of it, so no other rule is tried. It is
the one test candidate. The real rule hid 77 values, of 96 old ones. Also reported on test, for information: the same design with
nothing hidden.

## Step 2: why one sentence fails (dev only, 4 Nov 2026, about 02:00 IST, while the amendment 7 test runs)

With the notebook teacher, on 16 dev people, table + sharp: of 80 plain questions, 41 are right, 39 pick another fact of the same
person, and none picks an outsider. Other words: 33 right, 46 own other fact, 1 outsider. So the Brain knows who, and confuses
which kind of fact. Friend and pet do worst; toy does best (15 of 16).

At the first letter, the right memory is closer than the person's other offered facts on 79% of plain and 75% of other-words
questions, against 51% and 41% right answers. So part of the loss is in the read, not the address.

**A weakness in my reading of the exam:** the never-changed questions offer only outsiders as wrong options, never the person's
own other facts. So their high score shows the Brain knows who, not that it can tell a person's facts apart. Only the one-sentence
questions test that, and it is the score that is low.

**Next, on dev only:** a read that follows one memory. The closest memory at the first letter is chosen, and the rest of the name
follows that memory. Also sharp settings with a wider `near`. Any candidate gets its plan and predictions here before a test on
fresh people.
