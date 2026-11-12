"""Verify the released benchmark, embedded Kaggle prompts and saved paired results."""
import ast
import base64
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from math import comb
from pathlib import Path
import random
from zipfile import ZipFile

from jaxalyzer import analyze_packet
from jaxalyzer.hints import HEADER

ROOT = Path(__file__).resolve().parent


def load(path):
    return json.loads((ROOT / path).read_bytes())


def notebook_prompts(arm):
    folder = ROOT / "notebooks" / arm
    setup = load(folder / "kernel-metadata.json")
    notebook = load(folder / setup["code_file"])
    packages = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            for node in ast.walk(ast.parse("".join(cell["source"]))):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("UEsDB"):
                    packages.append(node.value)
    wheel, = packages
    with ZipFile(BytesIO(base64.b64decode(wheel, validate=True))) as archive:
        records = archive.read(f"jax_diagnosis_paper/data/{arm}/requests.jsonl")
    requests = [json.loads(line)["messages"] for line in records.splitlines()]
    assert all(len(messages) == 1 and messages[0]["role"] == "user" for messages in requests)
    return Counter(messages[0]["content"] for messages in requests)


def paired_stats(rows, case_groups, as_run=False):
    wins = losses = baseline_correct = assisted_correct = 0
    blocks = {}
    for row in rows:
        baseline = row["baseline"]["grade"] == "correct"
        arm = row.get("as_run", row["assisted"]) if as_run else row["assisted"]
        assisted = arm["grade"] == "correct"
        baseline_correct += baseline
        assisted_correct += assisted
        wins += assisted and not baseline
        losses += baseline and not assisted
        block = blocks.setdefault(case_groups[row["case"]], [0, 0])
        block[0] += int(assisted) - int(baseline)
        block[1] += 1
    discordant = wins + losses
    prob = min(1.0, 2 * sum(comb(discordant, i) for i in range(min(wins, losses) + 1)) / 2**discordant)
    rng = random.Random(20261111)
    samples = []
    for _ in range(5000):
        draw = rng.choices(list(blocks.values()), k=len(blocks))
        samples.append(sum(a for a, _ in draw) / sum(b for _, b in draw))
    samples.sort()
    return {"baseline": baseline_correct, "assisted": assisted_correct, "wins": wins, "losses": losses,
            "paired_p": prob, "owner_bootstrap_95": [samples[124], samples[4874]]}


def main():
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    pairs = load("benchmark/released-pairs.json")
    roster = load("benchmark/cases.json")
    hints = load("benchmark/reviewed-hints.json")
    results = load("evaluation/gemma.json")
    groups = {r["case"]: r["group"] for r in roster if r["eligible"]}
    assert set(groups) == set(hints) == {r["case"] for r in pairs} == {r["case"] for r in results}
    assert len(pairs) == len(results) == len(groups) == 90 and len(roster) == 92
    assert len(set(groups.values())) == 78
    for arm in ("baseline", "analyzer"):
        assert notebook_prompts(arm) == Counter(r[arm] for r in pairs), arm
    exact_raw = 0
    for row in pairs:
        flat = row["baseline"]
        reviewed = analyze_packet(flat, hints[row["case"]])
        assert row["analyzer"] == flat + HEADER + reviewed["evidence"], row["case"]
        assert reviewed["method"] == row["mode"]
        raw = analyze_packet(flat)
        exact_raw += bool(raw and raw["evidence"] == reviewed["evidence"])
    final = paired_stats(results, groups)
    fresh = paired_stats(results, groups, as_run=True)
    assert (final["baseline"], final["assisted"], fresh["assisted"], exact_raw) == (25, 38, 37, 50)
    probes = [r["fact_probe"] for r in results if "fact_probe" in r]
    assert len(probes) == 43
    report = {"coverage": [len(pairs), len(roster)], "reviewed_exact": len(pairs), "raw_exact": exact_raw,
              "as_run": fresh, "released": final, "fact_transitions": {
                  label: dict(Counter(p["new"] for p in probes if p["baseline"] == label))
                  for label in ("conflicting", "partial", "full")}}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
