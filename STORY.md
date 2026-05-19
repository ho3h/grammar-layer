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

These are all *content* concepts. There is no concept in GPT-2's top suppressors that corresponds to "the verb to be" or "the word is". Same prompt, same vocabulary (we checked — GPT-2 has features for "is" and for the copula in its dictionary), and yet **the suppression layer in GPT-2 is made of content concepts, not grammar concepts**.

This is the central finding. It is not that GPT-2 lacks the vocabulary, or fails to recognise the pattern, or doesn't have any concept of grammar. It's that GPT-2's prediction machinery does not include a *coordinated grammar-suppression apparatus*. Its predictions are made by content concepts pushing for and against each other. Gemma's predictions are made by content concepts pushing for, and grammar concepts pushing against.

The same asymmetry holds across a 52-prompt benchmark spanning capitals, weekdays, math, named entities, syntactic continuations, factual recall, instruction-following, coding, multi-step arithmetic, pronoun resolution, and summarisation. Gemma's opposing features are about **3× more likely to be grammar-flavoured** than its supporting features. On capital-completion prompts specifically, the ratio is **16×**. In GPT-2 the ratio is roughly 1× — no inversion.

## The brain analogy

This pattern of "structural features that fire across many tasks" reminds neuroscientists of something called the **multi-demand network** — a set of brain regions (frontoparietal cortex, mostly) that activate during many different cognitive tasks because they handle the *structural* work of cognition: holding things in working memory, switching between rules, applying a rule once you've found it. Neuroscientists distinguish this from **task-positive networks**, which activate in patterns specific to the content of the task: visual processing for visual tasks, language regions for language tasks.

By that analogy, **Gemma has a multi-demand network for next-token prediction** — a structural backbone that participates across diverse tasks. Its job is not to provide the answer; its job is to enforce the grammatical defaults that the answer has to fight through. **GPT-2 looks more like task-positive networks all the way down** — content-specific machinery, no structural coordinator.

We did not expect to find this. The original framing of this project assumed that "predicate features" (concepts about verbs, copulas, and so on) would be load-bearing for the answer — that they would actively help the model say "Tokyo" by structuring the completion as "X is Y". The opposite is true. Predicate features are load-bearing for what the model *doesn't* say. They are a generative default, and the answer wins by overcoming them.

## What this implies

For people who think about language models in general:

- "Same vocabulary, same task, very different internal structure" is a real phenomenon. Don't reach for "GPT-2 is just dumber" as the explanation. There's a qualitative difference in the prediction architecture, not only a quantitative one.
- Capability is not the same thing as routing. Both models can ablate to the same content concepts; only one of them has the grammar layer fighting the answer.
- If you've heard the recent narrative about emergent structure in larger / newer models — the idea that more recent training recipes produce richer internal world-models — Gemma's grammar layer is a concrete example. We can't prove that Google's training recipe caused this (we don't have ablated-training-data versions of Gemma), but it's the most parsimonious explanation for why one 2B model has this and a 124M model from a different lineage doesn't.

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

The full numbers are in the technical writeup. The qualitative read: Gemma's prose is measurably more grammatical-template-shaped than GPT-2's on every metric. There is an unavoidable size confound (Gemma 2 2B is much bigger than GPT-2 small), so this should be read as suggestive evidence consistent with the internal finding rather than an isolated demonstration. The next step is a matched-size comparison against Pythia 2.8B once its SAE labels are populated, which will tell us whether the behavioral signature is Gemma-specific or scale-driven.

The behavioral test passes the smell test that a non-specialist can apply directly: read a few paragraphs from each model and check whether you can hear the difference. If you can, the internal finding has external skin.

## Reading the rest

The full technical writeup is at [reports/writeup_v3_revised.md](reports/writeup_v3_revised.md). It covers the cross-model breadth (seven SAE-equipped language models, from Pythia 70M to Gemma 2 9B), the per-category breakdown, the ablation methodology, the targeting controls, mean-ablation replication, capital-prompt fingerprint, behavioral signature, and the open follow-ups. The interactive viewer at [apps/grammar_layer/index.html](apps/grammar_layer/index.html) lets you walk the feature space of each model. The hero figure for the case study above is [reports/viz_smoking_gun.png](reports/viz_smoking_gun.png).

If you want a single sentence to take away: **Gemma 2 thinks in grammar and content at the same time; GPT-2 just thinks in content. The grammar layer is the part that has to be talked over to get the specific answer out.**
