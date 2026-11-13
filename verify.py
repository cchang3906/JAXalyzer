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


def paired_stats(rows, case_groups):
    wins = losses = baseline_correct = assisted_correct = 0
    blocks = {}
    for row in rows:
        baseline = row["baseline"]["grade"] == "correct"
        assisted = row["assisted"]["grade"] == "correct"
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
    eligible = {r["case"]: r for r in roster if r["eligible"]}
    groups = {case: row["group"] for case, row in eligible.items()}
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
    fresh_correct = sum(row.get("as_run", row["assisted"])["grade"] == "correct" for row in results)
    assert (final["baseline"], final["assisted"], fresh_correct, exact_raw) == (25, 38, 37, 50)
    families = {}
    transitions = Counter()
    for row in results:
        family = eligible[row["case"]]["family"].split()[0].removeprefix("shape-")
        counts = families.setdefault(family, {"cases": 0, "baseline": 0, "assisted": 0})
        counts["cases"] += 1
        before, after = (row[arm]["grade"] for arm in ("baseline", "assisted"))
        counts["baseline"] += before == "correct"
        counts["assisted"] += after == "correct"
        if before != after:
            transitions[f"{before} -> {after}"] += 1
    usage = {}
    for arm in ("baseline", "assisted"):
        usage[arm] = {token: sum(row[arm]["usage"][token] for row in results) / len(results)
                      for token in ("prompt_tokens", "completion_tokens")}
    probes = [r["fact_probe"] for r in results if "fact_probe" in r]
    assert len(probes) == 43
    labels = {"conflicting": 0, "partial": 1, "full": 2}
    shifts = [labels[p["new"]] - labels[p["baseline"]] for p in probes]
    runs = load("evaluation/runs.json")
    pilot = load("evaluation/luna-pilot.json")["all"]
    report = {
        "evaluated_cases": len(results), "source_groups": len(set(groups.values())),
        "released": final, "mean_tokens_per_case": usage, "by_family": families,
        "grade_changes": dict(transitions),
        "source_facts": {"cases": len(probes), "improved": sum(d > 0 for d in shifts),
                         "worsened": sum(d < 0 for d in shifts), "unchanged": shifts.count(0),
                         "baseline": dict(Counter(p["baseline"] for p in probes)),
                         "assisted": dict(Counter(p["new"] for p in probes))},
        "timed_jobs": {"baseline_seconds": runs["baseline"]["seconds"],
                       "fresh_assisted_seconds": runs["expanded_as_run"]["seconds"],
                       "fresh_assisted_correct": fresh_correct},
        "luna_pilot": {k: pilot[k] for k in ("n", "flat_correct", "analyzer_correct")},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
