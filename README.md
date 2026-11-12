# JAXalyzer

JAXalyzer derives source relationships for JAX debugging questions. It reads a UTF-8 packet containing the application source and a recorded traceback and returns evidence about calls, operands, origins and tensor axes. It uses the Python standard library and does not execute the supplied program.

## Run locally

Use Python 3.11 from the repository root.

```sh
python -m jaxalyzer path/to/packet.txt
python -m unittest discover -s tests
python verify.py
```

The packet format and 90 complete examples are in [released-pairs.json](benchmark/released-pairs.json). Pass a row’s `baseline` text to the CLI. The output is JSON with `method` and `evidence`, or `null` on abstention. From Python, `jaxalyzer.analyze_packet(text, reviewed_hint=None)` also accepts an existing reviewed relationship. The [reviewed hints](benchmark/reviewed-hints.json) reproduce all 90 published prompts exactly. Raw packets alone reproduce 50 of those prompts exactly.

## Reproduce the Gemma comparison

Each notebook contains its inference wheel and prompt manifest. Use the accompanying Kaggle metadata for the pinned model, image and wheelhouse. Each arm takes one generation per case, with four NVIDIA L4 GPUs, no network during inference, temperature zero and a 768-token output cap. The notebooks require access to their listed Kaggle inputs. The model weights and third-party wheelhouse are hosted separately.

- [Baseline notebook](notebooks/baseline/gemma-12b-jax-90-baseline.ipynb) and [Kaggle configuration](notebooks/baseline/kernel-metadata.json)
- [Released analyzer notebook](notebooks/analyzer/gemma-12b-reviewed-source-evidence.ipynb) and [Kaggle configuration](notebooks/analyzer/kernel-metadata.json)

The verification command checks the file hashes, reads both embedded request manifests without launching inference, rebuilds the reviewed prompts, and recomputes the paired accuracy results from the saved grades.

## Benchmark and evaluation

| File | Contents |
| --- | --- |
| [Cohort and keys](benchmark/cases.json) | All 92 diagnoses, owner labels, families, frozen reference keys, evidence eligibility, and the two excluded packets |
| [Paired model inputs](benchmark/released-pairs.json) | Exact baseline and assisted prompt text for the 90 evaluated cases |
| [Evidence review](benchmark/reviews.json) | Earlier source and key decisions and final source-audit outcomes indexed by case |
| [Grading rubric](benchmark/RUBRIC.md) | Requirements for a correct, partial or incorrect diagnosis |
| [Gemma results](evaluation/gemma.json) | Paired answers, blind grades and reasons, token counts, the two original pre-fallback outputs, and the 43 source-fact probes. Join to cohort metadata by `case` |
| [Run receipts](evaluation/runs.json) | Gemma model, hardware, timed batches, archived export digests and prompt-manifest digests |
| [Luna pilot](evaluation/luna-pilot.json) | Earlier 17-case pilot with its own evidence version and denominator |

On this previously studied cohort, the baseline scored 25/90 fully correct and the reviewed analyzer arm scored 38/90, with 15 gains and two losses. The fresh analyzer job scored 37/90. Two source-review rejections caused fallback to earlier reviewed relationships and their archived answers to the exact prompts. The released comparison combines 88 new answers with these two saved answers. The 78 owner labels describe source clustering. The results are retrospective.

## Source material

Packets retain their source references and any included attribution, licenses and reconstruction notices. Source code, issue excerpts, course material and notebook reconstructions can carry different upstream terms. In particular, NB-048 and NB-049 identify unresolved coursework provenance and use factual reconstructions. The contents of this repository do not grant new rights to third-party material.
