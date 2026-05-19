# Two models, same question, very different inside

*A non-technical walkthrough of the finding. The full methods are in [reports/writeup_v3_revised.md](reports/writeup_v3_revised.md).*

---

Ask a language model to finish the sentence "**The capital of Japan is**". Two different models, two very different stories about what happens inside them.

**Gemma 2 (a 2-billion-parameter model from Google) answers "Tokyo."**

**GPT-2 small (a 124-million-parameter model from OpenAI, released in 2019) doesn't get it.** It thinks the most likely next word is "the".

Why does Gemma get it right and GPT-2 not? The boring answer is "Gemma is bigger and more recent". The interesting answer is what happens when you open them up and look at the actual machinery producing the prediction.

## How to look inside a language model

A language model doesn't store knowledge the way a database does. There's no row in a table that says "Japan → Tokyo." Instead, the model has thousands of internal *concepts* that activate in response to different patterns in the input — things like "this is a question about geography", "this is a place that has been called a capital", "this is text about a famous person", and so on. When the model writes the next word, all of these concepts are contributing simultaneously, each tugging the prediction in a slightly different direction.

A technique called a **sparse autoencoder** (SAE) lets us see which concepts are active for a given input and how much each one matters. Researchers then write down what each concept means by looking at the inputs that light it up. (We didn't generate these labels; they come from a public service called Neuronpedia that has labelled millions of these concepts.)

For our purposes you can think of it as putting electrodes on the model: we can read which "neurons" (really, conceptual clusters) are firing for any given prompt, and we can also *silence* specific ones and see what happens to the prediction.

## What's lighting up inside Gemma when it says "Tokyo"

The five most important concepts pushing Gemma toward "Tokyo" are, roughly:

1. *References to significant documents or publications*
2. *References to churches, bishops, and geographical locations in religious contexts*
3. *Interrogative and rhetorical questions about historical events or entities*
4. *Entities related to prominent figures and titles in Japanese history*
5. *References to programming concepts and technical terms*

These aren't obvious. None of them is "Japan-concept" or "capital-concept" in any clean sense. They're all *adjacent* to the answer — Tokyo is mentioned in significant documents, in geographical contexts, in questions about history, in lists of Japanese figures. Each concept contributes a partial pointer; together, they pin down "Tokyo".

If we silence these five concepts at the moment Gemma is about to write the word, **Gemma stops saying "Tokyo" and starts saying "a"**. The probability of "Tokyo" drops from about 17% to less than 0.2% — a roughly 100× collapse. The model still knows everything it knew before, in the sense that nothing else has been changed. But the specific routing structure that lifted "Tokyo" out of the noise is gone, and the model defaults to the generic completion "the capital of Japan is *a* city / *a* country / *a* place".

This is not what happens if you silence ten random concepts. We ran the obvious control across all 52 prompts in the benchmark: silence ten randomly-chosen active concepts instead of the targeted ten, average over five different random selections, and measure the same effect. The targeted ablation reduces target probability by about 3.5 nats on average (roughly a 30× drop). The random ablation reduces it by less than 0.1 nats. **The targeted set is doing about 40× more causal work than a random set of the same size.** Even more strikingly, if you silence the *ten least relevant* active concepts (the ones the attribution analysis says contribute almost nothing), there is no effect at all — Δlog P of essentially zero across every category. The model is not just "fragile to ablating ten features"; it is specifically routing through the features we identified.

## The surprise: Gemma is fighting against itself

Here's the part we didn't expect. While those five concepts are pushing Gemma toward "Tokyo", there is a *second* set of concepts that are pushing it *away* from "Tokyo" — actively suppressing the specific answer. The five top suppressors are:

1. *Past and present tense forms of the verb "to be"* — Gemma feature #15596
2. *Sociopolitical issues and violence in different regions*
3. *Instances of the word "is" in various contexts* — Gemma feature #10142
4. *References to countries and their significant characteristics*
5. *References to socio-economic conditions*

Look at items 1 and 3. The single most powerful concept *opposing* the answer is the concept of the verb "to be". The third most powerful is the concept of the word "is" itself. These aren't content concepts — they're **grammar concepts**.

And here's what we hadn't expected at all: **those same two features — #15596 and #10142 — show up as top opposers in every single capital-completion prompt we tested**. "The capital of France is...", "The capital of Germany is...", "The capital of Italy is...", "The capital of Spain is...", "The capital of Russia is...", "The capital of Japan is..." — six different prompts, six different correct answers (Paris, Berlin, Rome, Madrid, Moscow, Tokyo), and the same two grammar features are actively suppressing the specific capital in all six. They are not a per-prompt coincidence; they are a *fingerprint* — the same coordinated suppression apparatus, firing the same way, across every "X is Y" capital prompt.

What's happening is this: when Gemma reads "The capital of Japan is", it correctly recognises a familiar *grammatical pattern* — "X is Y" — and starts preparing the kinds of completions that pattern usually licenses. Most completions of "X is Y" in English text are generic: *Japan is a country, France is a republic, Mars is a planet*. The grammar concepts are pulling Gemma toward those generic continuations. The content concepts are pulling Gemma toward the specific answer "Tokyo". The specific answer wins, but only just — by about half a nat of log-probability, which is a fraction of a bit of information.

In other words: **Gemma's grammar machinery is actively suppressing the right answer.** To say "Tokyo", Gemma has to fight its own structural intuitions.

## GPT-2 has no such fight

Now ask GPT-2 the same prompt. GPT-2 doesn't know the answer — it would guess "the" if you let it. But if we run the same analysis on GPT-2, here's what we find pushing the model *away* from "Tokyo" (in case it were considering it):

1. *Names of famous individuals*
2. *Countries and locations*
3. *Words related to politics and government institutions*
4. *Information related to different countries and regions*
5. *Phrases containing names of organizations or companies*

These are all *content* concepts. There is no concept in GPT-2's top suppressors that corresponds to "the verb to be" or "the word is". Same prompt, same vocabulary — we checked, directly. GPT-2's sparse-autoencoder dictionary contains 652 features that match a strict grammar keyword classifier, including features explicitly labelled "the verb 'is' followed by descriptions or statements" and "instances of the word 'are'". None of these 652 features appear in GPT-2's top opposers on any of the six capital prompts. Not one.

**Same vocabulary, completely different routing.** Gemma recruits two specific grammar features to suppress every capital completion; GPT-2 owns hundreds of grammar features, recruits zero of them.

## Same finding shows up in other models too — it's not Gemma-specific

When we extended this analysis to other language models that have been instrumented with sparse autoencoders, the grammar-suppression pattern showed up in three out of five labelled models:

| Model | Organization | Parameters | Grammar enrichment on the opposing side |
|---|---|---|---|
| Pythia 70M | EleutherAI | 70 million | **5.8×** — fingerprint includes a "verb is" feature |
| GPT-2 small | OpenAI | 124 million | 0.93× — no enrichment |
| Gemma 1 2B | Google (2024) | 2 billion | **3.4×** — three "verb is" features in the fingerprint |
| Gemma 2 2B | Google (2024) | 2 billion | **2.8×** — the original (15596, 10142) fingerprint |
| Gemma 2 9B at mid-network layer 20 | Google (2024) | 9 billion | 1.3× — mild; likely a layer-pick artifact |

The most striking entry in this table is the first one. **Pythia 70M** — a tiny 70-million-parameter model from EleutherAI, built and trained with a completely different recipe from Gemma — has the strongest grammar-suppression enrichment of any model we tested. Its rank-1 cross-prompt opposer on capital completions is literally labelled *"occurrences of the verb 'is' and its various forms"*. The same coordinated suppression of specific completions in favor of grammatical "X is Y" defaults appears in EleutherAI's open-source Pythia.

This means the grammar layer is **not a Gemma-2 fingerprint**, not a Google fingerprint, and not a scale fingerprint. It is something more like a *family-and-depth* phenomenon — present at certain layers in certain training lineages. The interesting open question is now "why does GPT-2 small lack this, when other small models have it?" rather than "why does Gemma have this".

## The brain analogy

This pattern of "structural features that fire across many tasks" reminds neuroscientists of something called the **multi-demand network** — a set of brain regions (frontoparietal cortex, mostly) that activate during many different cognitive tasks because they handle the *structural* work of cognition: holding things in working memory, switching between rules, applying a rule once you've found it. Neuroscientists distinguish this from **task-positive networks**, which activate in patterns specific to the content of the task: visual processing for visual tasks, language regions for language tasks.

By that analogy, **Gemma has a multi-demand network for next-token prediction** — a structural backbone that participates across diverse tasks. Its job is not to provide the answer; its job is to enforce the grammatical defaults that the answer has to fight through. **GPT-2 looks more like task-positive networks all the way down** — content-specific machinery, no structural coordinator.

We did not expect to find this. The original framing of this project assumed that "predicate features" (concepts about verbs, copulas, and so on) would be load-bearing for the answer — that they would actively help the model say "Tokyo" by structuring the completion as "X is Y". The opposite is true. Predicate features are load-bearing for what the model *doesn't* say. They are a generative default, and the answer wins by overcoming them.

## What this implies

For people who think about language models in general:

- "Same vocabulary, same task, very different internal structure" is a real phenomenon. Don't reach for "GPT-2 is just dumber" as the explanation. There's a qualitative difference in the prediction architecture, not only a quantitative one.
- Capability is not the same thing as routing. Both models can ablate to the same content concepts; only one of them has the grammar layer fighting the answer.
- The grammar layer isn't a scale effect. **Pythia 70M — a tiny EleutherAI model 18× smaller than GPT-2 — has it stronger than any other model we measured.** It's not Google's recipe either. So the right question to ask isn't "what made some models special enough to grow a grammar layer" but "what feature of GPT-2 small's particular layer-and-training stopped it from having one". Why does GPT-2 small route differently from its tiny EleutherAI peer at the same scale?

For people who think about interpretability specifically:

- The original "predicate backbone" framing — pick features that participate in many attribution circuits, call them load-bearing — turns out to be exactly backwards. Those features are coordinated, and they are causally relevant, but their role is **suppression**, not **promotion**. Confusing breadth-of-participation with depth-of-effect cost this project a re-framing.
- The right test is per-prompt: rank features by their signed contribution to the target, look at the top-supporting and top-opposing sets separately. The interesting structure lives on the opposing side.

## Does the internal finding show up in what each model writes?

So far we've been looking inside both models with a kind of microscope — silencing concepts, watching predictions move. The natural next question: if Gemma really has a grammar layer that biases its predictions toward "X is Y" templates, does that show up in the prose it generates? Are Gemma's paragraphs *more grammatical, more copular, more hedged* than GPT-2's?

We tested this directly. Fifteen open-ended prompts (story openings, instructions, factual synthesis, conversational), five different sampling runs per prompt, three hundred tokens of generation each — 75 paragraphs from each model. Then we measured four behavioral signatures that the v3 finding *predicts* should be more frequent in Gemma:

- How often each model uses forms of the verb "to be" — *is, are, was, were, been, being*.
- How often it uses hedge / modal words — *may, might, could, would, perhaps, generally, typically*.
- How often it constructs generic noun phrases like "a thing", "the way", "a kind".
- What fraction of sentences open with copula-led structures like "This is…", "There are…", "It was…".

The results: Gemma's continuations are higher than GPT-2's on all four metrics. The two largest differences — copula-led sentence openers (5% vs 2%, p = 0.018) and hedge density (1.85 vs 1.16 per 100 tokens, p = 0.050) — reach statistical significance with just 75 generations per model. Copula density and generic-NP rate trend in the same direction but don't pass with this sample size. **Four out of four metrics point the way the internal finding predicts; two of the four are significant.**

There is an unavoidable size confound — Gemma 2 2B is 16× larger than GPT-2 small. The clean control is the comparison with Pythia 70M (which has the grammar layer per the internal analysis, but is *smaller* than GPT-2). If Pythia's prose shows the same grammatical signature as Gemma's despite being so much smaller than GPT-2, the behavioral signature decouples from scale entirely. That comparison is in flight as of writing and is the most informative remaining test.

The behavioral test passes the smell test that a non-specialist can apply directly: read a few paragraphs from each model and check whether you can hear the difference. Three examples from the same prompt, "Climate change is":

> **Gemma 2 2B:** *"is a reality. Our planet is warming, and the Arctic is melting. The United Nations' Intergovernmental Panel on Climate Change released a report in October 2019 that showed that the climate is warming at a rate that has never been seen before. The Arctic is melting at an alarming rate…"*

> **Pythia 70M:** *"is a major problem in the region, and it is a major cause of climate change. There are many other factors that can affect the dynamics of the climatic systems, such as the rate of change of the climate, the average daily temperature of the regions…"*

> **GPT-2 small:** *"occurring at a rate of nearly 1,000 times faster than the global average, according to a new study by scientists from the University of California at San Francisco. 'The planet is changing at a rate of about 1,000 times faster than the global average…'"*

Both Gemma and Pythia start with "is a X" — they take the copular template the grammar layer was set up for. GPT-2 doesn't; it continues into "occurring at a rate of…", a participial phrase that bypasses the X-is-Y template entirely. Pythia is the most striking — it's smaller than GPT-2 small, was trained on different data with a different recipe, and yet its prose has the same copular shape as Gemma's. That's the signature of the internal grammar layer working through to surface behavior.

## Reading the rest

The full technical writeup is at [reports/writeup_v3_revised.md](reports/writeup_v3_revised.md). It covers the cross-model breadth (seven SAE-equipped language models, from Pythia 70M to Gemma 2 9B), the per-category breakdown, the ablation methodology, the targeting controls, mean-ablation replication, capital-prompt fingerprint, behavioral signature, and the open follow-ups. The interactive viewer at [apps/grammar_layer/index.html](apps/grammar_layer/index.html) lets you walk the feature space of each model. The hero figure for the case study above is [reports/viz_smoking_gun.png](reports/viz_smoking_gun.png).

If you want a single sentence to take away: **Gemma 2 thinks in grammar and content at the same time; GPT-2 just thinks in content. The grammar layer is the part that has to be talked over to get the specific answer out.**
