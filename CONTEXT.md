# ClawDPO

ClawDPO governs autonomous preference-optimization campaigns that turn observed model failures into evaluated candidate models. Its responsibility ends before production release or deployment.

## Language

**Accepted Candidate**:
A candidate model that does not regress on Test Set factual correctness and that Codex judges clearly better than the Best Model under the shared Evaluation Specs. Ties and uncertain comparisons are not accepted, and acceptance never means deployment.
_Avoid_: Released model, production model

**Dataset Revision**:
An immutable, archived version of the preference-training data used by one or more Training Iterations.
_Avoid_: Latest data, current data folder

**Asynchronous Preference Improvement**:
The RL-like loop in which a Behavior Policy produces replies, external stages turn correctness and quality feedback into Preference Pairs, and DPO learns the next policy. It is preference-based policy improvement, not trajectory-level reinforcement learning.
_Avoid_: Static offline DPO, full RL

**Behavior Policy**:
The exact model revision that generates a Rollout Set and scores the Policy Likelihood used for the next policy update. Every response also retains the model revision that originally produced its text.
_Avoid_: Latest model, unspecified model

**Policy Lag**:
The revision distance between the policy that supplies a response's eligibility likelihood and the policy being updated from it. The first version requires zero scoring lag; a historical response may be reused only after the current Behavior Policy rescored it.
_Avoid_: Stale likelihood, unscored historical response

**Correctness Gate**:
The fast automated first-pass check that applies the shared Evaluation Specs to factual correctness and disqualifying failures such as hallucination. Passing makes a reply eligible for Codex review; failing disqualifies it from ever being chosen.
_Avoid_: Quality ranker, final pair selector

**Response Quality**:
The ordered judgment of how well an acceptable reply answers the prompt. Codex applies it when constructing Preference Pairs, while the Pairwise Quality Judge applies it when comparing a Candidate Model with the Best Model.
_Avoid_: Correctness, machine-evaluation pass

**Pairwise Quality Judge**:
The fast automated comparator that applies the shared Evaluation Specs to decide which of two replies to the same Test Set prompt is better. It provides a first pass before the stronger Codex judgment and remains separate from the Correctness Gate.
_Avoid_: Correctness evaluator, scalar reward

**Evaluation Specs**:
The two owner-provided Markdown files that define response-quality comparison and factual-error detection. Curl evaluators use them for fast screening and Codex receives the same files as context for stronger final judgment, so ClawDPO does not invent a second rubric.
_Avoid_: Generated rubric, separate Codex criteria

**Best Model**:
The strongest retained model revision. It is replaced only when a Candidate Model has no factual-correctness regression and Codex judges it clearly better on the same Test Set; ties and uncertainty preserve the existing Best Model.
_Avoid_: Latest checkpoint, production model

**Policy Likelihood**:
The current Behavior Policy's raw token logprobs for a response. Mean token logprob is the operational ranking score within one prompt's Rollout Set, cumulative logprob is retained for audit, and historical replies are rescored under the current policy before reuse.
_Avoid_: Quality score, correctness confidence

**Practical Policy Support**:
The replies that are empirically plausible under the current Behavior Policy's standard raw rollout, rather than every sequence with merely nonzero mathematical probability. Both sides of a Preference Pair must lie within this support; observing a reply once is evidence but does not automatically admit an isolated Extreme-Tail outlier.
_Avoid_: Any syntactically possible reply, fabricated negative

**Supported Tail**:
The 5th-to-25th percentile of mean raw token logprob within one prompt's Rollout Set. A correct, high-quality reply from this region is a useful exploratory chosen because the current policy can reach it without producing it routinely.
_Avoid_: Lowest possible likelihood, off-support answer

**Extreme Tail**:
The bottom 5 percent of mean raw token logprob within one prompt's Rollout Set. Replies here are archived but are ineligible for the current Preference Pair.
_Avoid_: Supported Tail, ideal chosen

**Likelihood Distribution**:
The distribution of mean raw token logprob across one prompt's complete 256-response Rollout Set. The top 25 percent is high likelihood, the 5th-to-25th percentile is the Supported Tail, the bottom 5 percent is the Extreme Tail, and the middle 50 percent is not prioritized.
_Avoid_: Global probability threshold, cross-prompt ranking

**Candidate Slice**:
The bounded, deliberately varied subset selected from the current Rollout Set for Codex to inspect together with the complete Chosen History. It is a review budget for current responses, not a cap on historical context or a closed list of allowed Preference Pair patterns.
_Avoid_: Full rollout dump, training pair

**Retrieval Quadrant**:
One of the four indexing regions formed by Correctness Gate pass/fail and high/low Policy Likelihood. The regions help build a varied Candidate Slice; they do not define an exhaustive taxonomy of Preference Pairs, and historical provenance remains a tag rather than a fifth class.
_Avoid_: Sample Class, pair type, training role

**Preference Pair**:
Any valuable ordering between two replies to the same prompt that Codex selects after reviewing the Candidate Slice and relevant history. Both replies must be within Practical Policy Support; the chosen must be clearly better and factually sound, while the rejected is normally a high-likelihood behavior worth suppressing. A chosen may come from the low-likelihood tail but never from outside current-policy support.
_Avoid_: Two scored samples, correctness-gate output

**Absolute Reject**:
A reply that fails the Correctness Gate and is therefore permanently disqualified from the chosen side. Being bad is not enough to make it training data: it must also lie within Practical Policy Support and form a valuable ordering with a better chosen.
_Avoid_: Relative Reject

**Relative Reject**:
A high-current-likelihood reply that passes the Correctness Gate but has lower Response Quality than the newly selected chosen reply. Any entry in Chosen History may become a Relative Reject after current-policy rescoring when a better reply is found.
_Avoid_: Machine-evaluation failure, Absolute Reject

**Chosen History**:
Every reply previously selected as chosen for one prompt, retained with its originating model revision. The complete history is shown during pair construction after every entry is rescored under the current policy.
_Avoid_: Only the latest chosen, permanent ground truth

**Quality Frontier**:
The highest-Response-Quality member of a prompt's Chosen History. It is the current quality benchmark but does not replace or hide the rest of the history.
_Avoid_: Permanent chosen, ground truth answer

**Prompt Pool**:
The growing archive of new and historical prompts eligible for rollout by later model versions. Membership does not guarantee that a prompt contributes a Preference Pair to every Training Iteration.
_Avoid_: Training pairs, current dataset

**Rollout Set**:
The complete 256 replies sampled from the current model for one prompt under a fixed sampling configuration. It is processed and archived in full before a Candidate Slice is shown to Codex.
_Avoid_: Adaptive rollout, partial rollout

**Test Set**:
The fixed evaluation set used after every Training Iteration. Its failures may guide retrieval of separate training conversations, but its own conversations never enter a Dataset Revision.
_Avoid_: Blind holdout, acceptance-only set

**Training Iteration**:
One cycle that regenerates Preference Pairs from the current model, runs DPO training, and compares the result against the current best model on the Test Set. A model becomes the next starting point only when it improves that comparison.
_Avoid_: Automatic promotion
