# What using Shishya Brain looks like

*Real output, 31 Oct 2026, on a 4-core CPU. `python -m brain.demo brain/out/runs/brain3 --out brain/DEMO.md`.*

- **The model is v3:** v2's trained network (12.8M parameters, children's stories, read as bytes) with v3's
  memory address ([docs/PLAN.md](../docs/PLAN.md), amendment of 31 Oct).
- **Nothing is trained here.** No gradients, no optimiser. Only its lasting synapses change, by its local rule,
  as it reads.
- **Dev names and dev stories only,** never the test stream.
- **It writes its own answers:** each answer below is the text it wrote after the question, letter by letter.
  The 1-in-5 choice is the life test's measure.

## 1. It writes (empty memory)

> Once upon a time, there was a big orange dog named Max. Max loved to play outside in the sun. One day, Max saw a small, red ball stuck in a tree. He wanted to get it, but he was scared. Max saw a little bird in the tree. The bird said, "I can help you get down from the tree." The bird got close to Max and said, "No

## 2. It reads, and keeps what it read

Story 1 (574 ms to read and store): *One day, a monkey was playing in the park. Naros's cat is called Fugez. Pimev lives in a town called Zifig. He was very happy. He saw a big fair with many fun games and yummy food. The monkey wanted t…*

Then 30 ordinary stories (429 ms each).
Then one more story with a fact in it: *…The secret word of Burof is Desen.…*

## 3. Asked in a fresh context (nothing before the question)

> **Naros's cat is called** …
>   with its memory: **Fugez.**   (empty memory: Saram.)
>   choosing among Fugez, Tasoz, Dotok, Nusup, Varin: 100% on Fugez

> **Pimev lives in a town called** …
>   with its memory: **Zifig.**   (empty memory: Samam.)
>   choosing among Zifig, Tasoz, Dotok, Nusup, Varin: 100% on Zifig

> **The secret word of Burof is** …
>   with its memory: **Desen.**   (empty memory: Satit.)
>   choosing among Desen, Tasoz, Dotok, Nusup, Varin: 100% on Desen

## 4. A correction (reward 3: stored 3 times as strongly)

It reads: *Naros's cat is called Danat.*

> **Naros's cat is called** …
>   with its memory: **Danat.**   (empty memory: Saram.)
>   choosing among Danat, Fugez, Tasoz, Dotok, Nusup: 100% on Danat

## 5. Forget on request

`forget("Naros's cat is called Danat.")`

> **Naros's cat is called** …
>   with its memory: **Saraz.**   (empty memory: Saram.)
>   choosing among Danat, Fugez, Tasoz, Dotok, Nusup: 9% on Danat

## 6. The same question in other words (its known weak spot)

> **The town where Pimev lives is called** …
>   with its memory: **Movis.**   (empty memory: Suvin.)
>   choosing among Zifig, Tasoz, Dotok, Nusup, Varin: 18% on Zifig


## The same demo with v2's memory

The same story and questions with v2 as trained (`python -m brain.demo brain/out/runs/brain2`):
- Sections 1–4 give the same answers.
- After `forget`, v2 wrote **"Sugez."**: it lost the first letter, and the rest of the old name was still stored.
  v3 writes **"Saraz."**, which is close to its empty-memory guess "Saram.".
- Asked in other words, v2 also failed: "Movis.", 27% on Zifig.

## What the demo does not show

The gap here is 30 stories. The life test asks after hundreds of sessions, with 260 facts in the stream. There,
on the dev stream, v3's memory recalls 99% of the main facts and v2's 80% ([RESULTS.md](RESULTS.md) section 5).
