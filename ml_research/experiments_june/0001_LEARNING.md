# Mechinterp 101 from Scratch: A Hands-On Introduction

This series is for people who have done some neural-net basics, maybe followed Karpathy-style transformer material, and now want to understand mechanistic interpretability by actually touching a real model.

The goal is not to start with a polished benchmark or a giant framework. The goal is to learn the basic scientific loop:

1. Pick a tiny behavior.
2. Observe internal activations.
3. Form a cautious hypothesis.
4. Intervene on the model.
5. Use controls to prove yourself wrong.
6. Only then decide what the result means.

We will use an interactive Python workflow with IPython, TransformerLens, SAELens, and Gemma-Scope SAEs. Scripts will come later. This first post is the map.

---

## 0. The One-Minute Vocabulary

### Activation

An activation is a vector produced inside the model while it processes text.

When a transformer reads a prompt, every layer creates internal states. Those states are not words. They are high-dimensional numerical vectors. Mechanistic interpretability is partly the art of asking: what information is represented in those vectors, and what does it do?

### Logit

A logit is the model’s raw score for a possible next token before softmax turns scores into probabilities.

If the model sees:

```text
JSON: {"name": "Ada", "age": 42
```

it may assign high logits to tokens like:

```text
,
}
},
```

A useful behavior metric is often a logit difference, such as:

```text
logit("}") - logit(",")
```

That tells us whether the model is leaning more toward closing the object or continuing it.

### Ablation

An ablation means removing or zeroing out part of the model’s internal computation and seeing what changes.

If you think a feature helps the model predict `}`, you can ablate that feature. If the model stops predicting `}`, that is causal evidence. If nothing changes, the feature probably was not necessary. If the opposite happens, your story was wrong or incomplete.

Ablation is not interpretation by itself. It is an intervention.

### Amplification

Amplification means making an internal feature stronger instead of removing it.

If amplifying a feature increases `logit("}") - logit(",")`, that feature may push toward closing the JSON object. If amplifying it decreases that contrast, it may push toward continuation or suppress closure.

### Control

A control is a matched comparison case designed to catch fake explanations.

For example:

```text
Target:  JSON: {"name": "Ada", "age": 42
Control: JSON: {"name": "Ada", "age": 42}
```

The target is incomplete. The control is complete. If a feature changes both equally, it is not specific to the missing-brace behavior.

### Feature

In this series, a feature usually means a direction found by a sparse autoencoder, or SAE.

A feature is not automatically a clean human concept. A feature is a candidate direction in activation space. You still have to test what it does.

### Sparse Autoencoder, or SAE

An SAE is a model trained to decompose dense neural activations into a sparse set of feature activations.

Instead of staring at one dense vector with thousands of dimensions, we encode it into a much larger vector where only a small number of features are active. This can make internal structure easier to inspect.

Important warning: SAE features are tools, not truth. A feature firing does not mean you know what the model is thinking.

### Direct Logit Attribution, or DLA

DLA asks: if this feature writes a vector into the residual stream, which output tokens does that vector directly push up or down?

For a feature decoder vector, we can project it through the model’s unembedding matrix and ask whether it points toward tokens like `}` or `,`.

DLA is useful for generating candidates. It is not proof. A feature can have positive DLA to a token and still fail a causal intervention.

### Patching

Patching means replacing part of the model’s internal activation with a modified version.

For example, we can:

1. Run the model normally.
2. Capture the residual stream at layer 12.
3. Encode it with an SAE.
4. Change one feature.
5. Decode it back.
6. Patch the modified vector into the model.
7. Rerun the model and measure the output.

This is how we move from “the feature is active” to “the feature does something.”

### Reconstruction patch

When using an SAE, decoding the SAE activations back into the residual stream is already an approximation.

So we need to compare feature ablation against an unchanged SAE reconstruction, not only against the raw model.

Bad comparison:

```text
ablated SAE patch vs raw model
```

Better comparison:

```text
ablated SAE patch vs unchanged SAE reconstruction patch
```

This separates the effect of removing the feature from the general reconstruction error of the SAE.

### Specificity gap

A specificity gap measures whether an intervention affects the target more than the control.

A simple version:

```text
specificity_gap = abs(target_effect) - abs(control_effect)
```

If the target changes and the control does not, that is interesting. If both change equally, the intervention is probably not specific.

---

## 1. What We Are Trying to Learn

This series is not about proving a grand theory of language models.

It is about learning the practical scientific discipline of mechanistic interpretability.

We want to answer questions like:

```text
When a model predicts a closing brace in JSON, where does that behavior come from?
Is there a feature that helps produce it?
Is the feature causal?
Is it specific to incomplete JSON?
Does it generalize across prompts?
```

A good beginner result is not:

```text
We found the JSON circuit.
```

A good beginner result is:

```text
We tested a candidate feature. It looked plausible under activation and DLA, but ablation showed it was not a clean close-brace writer.
```

That is still a real lesson.

---

## 2. The Core Workflow

Every experiment in this series follows the same loop.

### Step 1: Pick a tiny behavior

Start with a deterministic, inspectable behavior.

Good beginner tasks:

```text
JSON object closure
Python indentation
parentheses or bracket balancing
quote closure
simple arithmetic formatting
```

Bad beginner tasks:

```text
truthfulness
moral reasoning
political ideology
long-chain reasoning
personality
```

The behavior should be small enough that you can define a clear target token and foil token.

Example:

```text
Prompt: JSON: {"name": "Ada", "age": 42
Target token: }
Foil token: ,
Metric: logit("}") - logit(",")
```

### Step 2: Run the model and inspect predictions

Before interpreting internal activations, check the model’s actual next-token predictions.

If the model is not doing the behavior at all, there is nothing to explain yet.

We want to see whether the target token is actually competitive.

For example:

```text
,
}
},
}}
```

If `}` is near the top, the behavior exists.

### Step 3: Capture activations

Use TransformerLens to capture a specific internal activation, such as:

```text
blocks.12.hook_resid_post
```

This is the residual stream after layer 12.

The residual stream is the main communication channel of the transformer. Attention heads and MLPs read from it and write back into it.

### Step 4: Encode activations with an SAE

Use a Gemma-Scope SAE to convert dense residual activations into sparse feature activations.

Now instead of asking:

```text
What does this 2304-dimensional vector mean?
```

we ask:

```text
Which SAE features are active at the final token?
```

This gives candidate features to inspect.

### Step 5: Do not trust top activations

The loudest feature is not necessarily the causal feature.

A top feature might represent:

```text
numbers
formatting
names
syntax context
generic JSON-ness
age-related semantics
```

That does not mean it causes the model to output `}`.

Top activation is observation, not causation.

### Step 6: Add a matched control

Compare the incomplete prompt to a matched complete prompt.

Example:

```text
Target:  JSON: {"name": "Ada", "age": 42
Control: JSON: {"name": "Ada", "age": 42}
```

Now we can search for features that fire on the target but not the control.

A better candidate is not merely active. It is specifically active when the behavior matters.

### Step 7: Use DLA as a candidate filter

For active features, compute whether their decoder direction points more toward the target token or the foil token.

For example:

```text
DLA_diff = DLA_to("}") - DLA_to(",")
```

Then combine activation specificity with functional direction:

```text
score =
  (activation_target - activation_control)
  *
  (DLA_to_target - DLA_to_foil)
```

This is still not proof. It is a ranking method.

### Step 8: Intervene

Now change the feature and rerun the model.

Basic interventions:

```text
ablation:      set feature activation to 0
amplification: multiply feature activation by 2 or 4
```

Measure the output metric again:

```text
logit("}") - logit(",")
```

If ablating a feature lowers the target behavior and amplification raises it, that is good causal evidence.

If nothing happens, it is not causal.

If the sign goes the opposite direction, your interpretation was wrong or the feature participates in a more complex circuit.

### Step 9: Compare against reconstruction

When patching through an SAE, always compare against an unchanged SAE reconstruction patch.

Three measurements matter:

```text
raw model
unchanged SAE reconstruction patch
feature-modified SAE patch
```

The feature effect is:

```text
feature_effect =
  modified_patch_metric - reconstruction_patch_metric
```

not:

```text
modified_patch_metric - raw_model_metric
```

This avoids mistaking SAE reconstruction error for feature causality.

### Step 10: Check target versus control

A real candidate should affect the target more than the control.

Example of a good sign:

```text
Target effect:  -0.75
Control effect:  0.00
```

That means the feature affects the target prompt specifically.

But the direction still matters. If amplification pushes the metric down, the feature is not a target-token promoter. It may be a continuation feature, a suppressor, or part of a distributed mechanism.

---

## 3. The First Big Lesson: Interpretation Is Easy to Overclaim

A common beginner mistake:

```text
This feature’s top DLA tokens look like code tokens, so it is a syntax feature.
```

That is not valid.

Another common mistake:

```text
This feature fires strongly on my prompt, so it explains the behavior.
```

Also not valid.

Another:

```text
This feature has positive DLA to my target token, so it causes that token.
```

Still not valid.

The correct standard is stricter:

```text
The feature is active on target more than control.
Its decoder direction favors the target over the foil.
Changing the feature causally changes the output metric.
The effect is stronger on target than control.
The sign makes sense.
The result survives more than one prompt.
```

Until then, use cautious language.

Say:

```text
candidate feature
target-specific effect
weak causal evidence
directionally inconsistent
not a clean writer
```

Do not say:

```text
we found the circuit
the model thinks X
this feature means Y
```

---

## 4. The First Real Negative Result

A negative result is not failure.

Suppose we find a feature that looks promising:

```text
fires on incomplete JSON
absent on complete JSON
DLA favors } over ,
```

Then we ablate and amplify it.

If the feature does not increase `logit("}") - logit(",")`, then it is not a close-brace writer.

That is progress. We learned:

```text
Layer 12 contains target-specific features.
Some features affect the target prompt.
But this layer/SAE did not reveal a simple close-brace-promoting feature.
```

The next scientific move is not to force a story. The next move is to scan other layers, test more prompts, or change the target contrast.

---

## 5. What Makes This Different From Normal ML Engineering

Normal ML engineering often asks:

```text
Can I make the model perform better?
```

Mechanistic interpretability asks:

```text
How does the model perform this behavior internally?
```

That changes the workflow.

We are not training. We are not optimizing. We are not building a product pipeline first.

We are doing something closer to experimental physics or biology:

```text
observe
perturb
measure
control
revise hypothesis
```

The tooling can become complex later, but the first job is to learn the experimental loop.

---

## 6. Why IPython Is a Good Starting Point

IPython is an interactive Python shell.

It is like a better terminal REPL:

```text
tab completion
history
multi-line paste
%run script.py
live variables
fast iteration
```

Jupyter notebooks use IPython as their Python kernel. So IPython is not a toy. It is the same interactive execution style without the browser interface.

For early mechinterp, IPython is useful because you can:

```text
load the model once
load the SAE once
try small probes interactively
inspect tensors
rerun small blocks
avoid restarting heavy model loads
```

Later, polished scripts are better for reproducibility. But early on, IPython is a good microscope.

---

## 7. Tools Used in This Series

### PyTorch

The tensor library underneath the whole workflow.

### TransformerLens

A library for loading transformer models in a hookable form.

It lets us capture and patch internal activations.

### SAELens

A library for loading and using sparse autoencoders.

It lets us encode dense model activations into sparse feature activations and decode them back.

### Gemma-2-2B

A small but capable open-weight model that fits on a 16 GB GPU in reduced precision.

### Gemma-Scope

A collection of pretrained SAEs for Gemma models.

This gives us high-quality feature decompositions without training our own SAE.

### IPython

The interactive environment where we run small experiments before turning them into scripts.

---

## 8. The Series Plan

### Part 1: What Mechanistic Interpretability Is

We explain the difference between model behavior and model mechanism.

Key ideas:

```text
behavior
mechanism
activation
logit
intervention
control
```

### Part 2: Setting Up the Microscope

We install the environment and load a real model.

Key ideas:

```text
GPU memory
bfloat16
model weights
tokenizers
interactive REPL
```

### Part 3: First Forward Pass

We run a JSON prompt and inspect the model’s next-token predictions.

Key ideas:

```text
tokens
logits
top-k predictions
target token
foil token
logit difference
```

### Part 4: Capturing Internal Activations

We capture the residual stream at a chosen layer.

Key ideas:

```text
residual stream
layers
hook points
activation cache
final token position
```

### Part 5: Sparse Autoencoders

We load a Gemma-Scope SAE and encode residual activations.

Key ideas:

```text
SAE
feature activation
decoder vector
sparsity
reconstruction
```

### Part 6: Why Top Features Are Not Explanations

We inspect top active features and learn why this is only observational.

Key ideas:

```text
activation is not causation
feature labeling is hard
DLA can be noisy
avoid story-first interpretation
```

### Part 7: Matched Controls

We compare an incomplete JSON prompt against a complete JSON prompt.

Key ideas:

```text
target prompt
control prompt
activation difference
specificity
```

### Part 8: Direct Logit Attribution

We use decoder vectors to rank candidate features by their direct output direction.

Key ideas:

```text
unembedding
DLA
target-vs-foil contrast
candidate ranking
```

### Part 9: Ablation and Amplification

We modify feature activations and patch them back into the model.

Key ideas:

```text
ablation
amplification
patching
causal effect
```

### Part 10: Reconstruction Controls

We learn why SAE reconstruction itself is an intervention.

Key ideas:

```text
raw baseline
reconstruction patch
feature-modified patch
feature effect
```

### Part 11: Reading Negative Results

We interpret a feature that is target-specific but points the wrong way.

Key ideas:

```text
negative result
sign mismatch
distributed mechanism
upstream feature
not a clean writer
```

### Part 12: From One Prompt to a Real Experiment

We scale from one prompt to a small prompt family.

Key ideas:

```text
prompt set
robustness
specificity gap
layer sweep
feature packet
reporting results honestly
```

---

## 9. What Counts as Success?

A beginner success is not discovering a famous circuit.

A beginner success is being able to say:

```text
I chose a behavior.
I defined a metric.
I found candidate features.
I tested them causally.
I used controls.
I rejected overconfident interpretations.
I learned what the evidence actually supports.
```

That is the foundation.

Everything else comes later.

---

## 10. The Ethic of This Series

This series should be honest about uncertainty.

Mechanistic interpretability is full of tempting stories. The model predicts a token, a feature fires, the top tokens look meaningful, and suddenly we want to narrate the model’s “thoughts.”

That is the trap.

The discipline is to slow down.

Do not trust the story until the intervention survives controls.

The point of this series is not to make mechanistic interpretability look magical. The point is to make it feel like real experimental work.

Sometimes the result will be:

```text
Nothing happened.
```

Sometimes:

```text
The sign went the wrong way.
```

Sometimes:

```text
The control moved too.
```

Those are not failures. Those are the moments where the model teaches us that our first explanation was too simple.

That is the work.
