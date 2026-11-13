# JAXalyzer: JAX Dependency Mapping Tool for Agentic Debugging

JAXalyzer reads application source and a recorded JAX traceback. Its rules identify relationships such as which call is evaluated first or which tensor axes must match. The assisted Gemma prompt appends those relationships to the same question, source and traceback that the baseline receives. [Read the paper](reports/JAXalyzer-revised.md).

## Results

On 90 previously studied tasks across 78 source groups, one-pass Gemma 4 12B produced 25 fully correct diagnoses with the baseline prompt and 38 with the released evidence (27.8% to 42.2%). There were 15 gains and two losses. The evidence added 628 input tokens per case on average. An earlier Luna pilot improved from 10/17 to 14/17 on overlapping cases. The paper reports the family breakdown, changes in diagnosis grades, and the targeted analysis of 43 difficult cases.

The fresh assisted job scored 37/90 in 496.5 seconds. The released comparison uses earlier reviewed prompts and their saved answers for two cases flagged during source review. Only the original jobs were timed. Case-level grades and the run receipts are in [evaluation](evaluation/).

## Reproduce locally

Use Python 3.11 from the repository root. These commands use the standard library.

```sh
python -m unittest discover -s tests
python verify.py
python -m jaxalyzer path/to/packet.txt
```

The verifier checks hashes, validates both notebooks’ embedded prompts, regenerates the released evidence, and reports the paper’s paired accuracy, grade transitions, token usage, family counts, and source-fact results from the saved records. For a new packet, the CLI reads UTF-8 text and returns the evidence as JSON, or `null` if the analyzer abstains. To reproduce a published card exactly, use that case’s [reviewed relationship](benchmark/reviewed-hints.json) along with the baseline packet.

## Data and inference

- [Paired prompts](benchmark/released-pairs.json), [case keys and source groups](benchmark/cases.json), and [scoring rubric](benchmark/RUBRIC.md). The 90 eligible rows form the paper’s cohort. The roster preserves two additional screened rows.
- [Gemma answers and blind grades](evaluation/gemma.json), [Luna pilot](evaluation/luna-pilot.json), and [inference receipts](evaluation/runs.json). Gemma records also include the auditor’s reasons and the source facts assessed in Section 6.
- [Baseline notebook](notebooks/baseline/gemma-12b-jax-90-baseline.ipynb) and [JAXalyzer notebook](notebooks/analyzer/gemma-12b-reviewed-source-evidence.ipynb). Each folder contains Kaggle metadata identifying the model, image and wheelhouse. Each arm uses four NVIDIA L4 GPUs and one generation per case.

Packets retain upstream attributions, license notices and labels for factual reconstructions. Third-party material remains subject to its original terms. Coursework provenance remains unresolved for NB-048 and NB-049, which use factual reconstructions.
