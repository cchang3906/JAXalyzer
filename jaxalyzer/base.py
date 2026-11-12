"""Exploratory source/trace relationships. Standard library only; never executes packets."""

import ast
import re
import textwrap
from dataclasses import dataclass, field


@dataclass
class Line:
    code: str
    physical: int
    label: str
    number: int | None = None
    arrow: bool = False


@dataclass
class Function:
    name: str
    args: list[str]
    lines: list[Line]


@dataclass
class Frame:
    label: str
    name: str
    args: list[str] = field(default_factory=list)
    lines: list[Line] = field(default_factory=list)


def tree(code):
    code = textwrap.dedent(code).strip()
    for candidate in (code, "{" + code + "}"):
        try:
            return ast.parse(candidate)
        except (SyntaxError, ValueError):
            pass
    return None


def expr(node):
    return ast.unparse(node)


def nodes(code, cls):
    parsed = tree(code)
    return [n for n in ast.walk(parsed) if isinstance(n, cls)] if parsed else []


def callable_name(node):
    return expr(node).split(".")[-1] if isinstance(node, (ast.Name, ast.Attribute)) else ""


def assignments(lines):
    result = []
    for i, line in enumerate(lines):
        for size in range(1, min(12, len(lines) - i) + 1):
            chunk = lines[i:i + size]
            if any(b.number != a.number + 1 for a, b in zip(chunk, chunk[1:])):
                break
            parsed = tree("\n".join(x.code for x in chunk))
            if parsed:
                for node in parsed.body:
                    if isinstance(node, ast.Assign):
                        result.append((node.targets, node.value, chunk))
                    elif isinstance(node, ast.AnnAssign) and node.value is not None:
                        result.append(([node.target], node.value, chunk))
                break
    return result


def bindings(target, value):
    if isinstance(target, ast.Name):
        return [(target.id, value)]
    if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)):
        if len(target.elts) == len(value.elts):
            return [p for t, v in zip(target.elts, value.elts) for p in bindings(t, v)]
    return []


class Packet:
    def __init__(self, text):
        self.text = text
        self.raw = text.splitlines()
        self.sources = []
        self.frames = []
        self.trace_lines = []
        mode, label, current = "", "", None
        for i, original in enumerate(self.raw, 1):
            s = re.sub(r"<[^>]+>", "", original) if "<a href=" in original else original
            if re.match(r"^RECORDED TRACEBACK", s):
                mode, current = "trace", None
                continue
            if re.match(r"^SOURCE(?: CONTEXT| EXCERPTS| \(|$)", s):
                mode, current = "source", None
                continue
            if re.match(r"^(?:LOCAL FACTS|DERIVED RELATIONSHIPS)", s):
                mode = "ignored"
                continue
            if s.startswith("SOURCE ATTRIBUTION"):
                mode = "ignored"
                continue
            if mode == "trace":
                self.trace_lines.append(Line(s, i, "recorded traceback"))
                factual = re.match(r"^Cell In\[\d+\]:(\d+)(?: in ([\w.<>]+))?:\s*(.+)$", s)
                if factual:
                    current = Frame(s, factual[2] or "<module>")
                    current.lines.append(Line(factual[3], i, "recorded factual call path", int(factual[1]), True))
                    self.frames.append(current)
                    continue
                native = re.match(r'^\s*File "([^"]+)", line (\d+), in (.+)$', s)
                archived = re.match(r"^(?:File |Input |Cell |[~/]|<ipython).*?\bin ([\w.<>]+)(?:\((.*?)\))?\s*$", s)
                if native:
                    current = Frame(native[1] + ":" + native[2], native[3])
                    self.frames.append(current)
                    continue
                if archived:
                    current = Frame(s, archived[1], self.signature_args(archived[2] or ""))
                    self.frames.append(current)
                    continue
                if re.match(r"^(?:Input In|Cell In|<ipython-input-|/\S+ (?:in|Cell)|~\S+ in)", s):
                    current = Frame(s, "<module>")
                    self.frames.append(current)
                    continue
                numbered = re.match(r"^\s*(?:(-+>)\s*)?(\d+)\s+(.*)$", s)
                if numbered and current:
                    current.lines.append(Line(numbered[3], i, current.label, int(numbered[2]), bool(numbered[1])))
                elif native is None and current and re.match(r"^    \S", s) and not re.match(r"^\s*[~^]+\s*$", s):
                    if not re.match(r"^\s*(?:pval|recipe|primal|tangent)\s*=", s):
                        current.lines.append(Line(s[4:], i, current.label, arrow=True))
            elif mode == "source":
                numbered = re.match(r"^\s*(\d+)(?: {2}|\t)(.*)$", s)
                if numbered:
                    self.sources.append(Line(numbered[2], i, label, int(numbered[1])))
                elif s.strip() and not s.startswith("SOURCE "):
                    label = s.strip()
        self.functions = self.find_functions()

    @staticmethod
    def signature_args(s):
        if not s:
            return []
        try:
            f = ast.parse("def f(" + s + "):\n pass").body[0]
            return [a.arg for a in f.args.posonlyargs + f.args.args]
        except SyntaxError:
            return []

    def find_functions(self):
        functions = []
        for i, line in enumerate(self.sources):
            match = re.match(r"\s*(?:async )?def (\w+)\(", line.code)
            if not match:
                continue
            indent = len(line.code) - len(line.code.lstrip())
            body = [line]
            j = i + 1
            while j < len(self.sources):
                nxt = self.sources[j]
                if nxt.label != line.label:
                    break
                if nxt.code.strip() and len(nxt.code) - len(nxt.code.lstrip()) <= indent:
                    if not line.code.rstrip().endswith(":") and not any(x.code.rstrip().endswith(":") for x in body):
                        body.append(nxt)
                        j += 1
                        continue
                    break
                body.append(nxt)
                j += 1
            header = " ".join(x.code.strip() for x in body)
            sig = re.match(r"(?:async )?def \w+\((.*?)\)(?:\s*->.*?)?:", header)
            args = self.signature_args(sig[1]) if sig else []
            functions.append(Function(match[1], args, body))
        for targets, value, refs in assignments(self.sources):
            if isinstance(value, ast.Lambda):
                for target in targets:
                    if isinstance(target, ast.Name):
                        functions.append(Function(target.id, [a.arg for a in value.args.args], refs))
        return functions

    def error_line(self, name):
        return next((x for x in self.trace_lines if re.search(r"\b" + re.escape(name) + r":", x.code)), None)

    def function_for(self, frame, line):
        leaf = frame.name.split(".")[-1]
        matches = [f for f in self.functions if f.name == leaf or leaf == "<lambda>"]
        same = [f for f in matches if any(x.code.strip() == line.code.strip() for x in f.lines)]
        return same[0] if len(same) == 1 else matches[0] if len(matches) == 1 else None

    def latest_assignment(self, name, before=None, function=None):
        scope = function.lines if function else self.sources
        if before and function and before.physical < function.lines[0].physical:
            source_match = [x for x in function.lines if x.code.strip() == before.code.strip()]
            if not source_match:
                target_calls = {expr(c) for c in nodes(before.code, ast.Call)}
                source_match = [x for x in function.lines if target_calls.intersection(expr(c) for c in nodes(x.code, ast.Call))]
            if len(source_match) != 1:
                return None
            before = source_match[0]
        found = []
        for targets, value, lines in assignments(scope):
            if before and lines[-1].physical >= before.physical:
                continue
            for target in targets:
                for bound, val in bindings(target, value):
                    if bound == name:
                        found.append((val, lines))
        return found[-1] if found else None

    def unique_global_assignment(self, name):
        found = []
        inside = {x.physical for f in self.functions for x in f.lines}
        for targets, value, lines in assignments(self.sources):
            if lines[0].physical in inside:
                continue
            for target in targets:
                for bound, val in bindings(target, value):
                    if bound == name:
                        found.append((val, lines))
        return found[0] if len(found) == 1 else None

    def callers(self, frame, function):
        target = function.name if function else frame.name.split(".")[-1]
        args = function.args if function else frame.args
        result = []
        index = self.frames.index(frame)
        for caller in reversed(self.frames[:index]):
            for line in caller.lines:
                for call in nodes(line.code, ast.Call):
                    if callable_name(call.func) != target:
                        continue
                    if "." in frame.name and ".<locals>." not in frame.name and function is None:
                        continue
                    if any(isinstance(a, ast.Starred) for a in call.args):
                        continue
                    if args and args[0] in {"self", "cls"} and isinstance(call.func, ast.Attribute):
                        mapped = {args[0]: call.func.value, **dict(zip(args[1:], call.args))}
                    else:
                        mapped = dict(zip(args, call.args))
                    mapped.update({k.arg: k.value for k in call.keywords if k.arg})
                    result.append((mapped, line))
            if result:
                break
        return result


def cite(line):
    return {"flat_line": line.physical, "source": line.label, "source_line": line.number, "text": line.code.strip()}


def result(kind, tier, relationship, evidence, withheld):
    unique = {x.physical: x for x in evidence if x}
    return {"kind": kind, "tier": tier, "relationship": relationship,
            "evidence": [cite(x) for x in unique.values()], "withheld_reason": withheld}


def call_origin(p, frame, formal):
    links, evidence, current = [], [], formal
    visited = set()
    for _ in range(5):
        key = (id(frame), current)
        if key in visited:
            break
        visited.add(key)
        f = p.function_for(frame, frame.lines[-1]) if frame.lines else None
        candidates = p.callers(frame, f)
        if len(candidates) != 1 or current not in candidates[0][0]:
            break
        mapped, line = candidates[0]
        actual = mapped[current]
        links.append(f"`{current}` <- `{expr(actual)}`")
        evidence.append(line)
        if not isinstance(actual, ast.Name):
            return links, evidence, actual
        owners = [fr for fr in p.frames if any(x is line for x in fr.lines)]
        if len(owners) != 1:
            break
        frame, current = owners[0], actual.id
        f = p.function_for(frame, line)
        if f and current not in f.args:
            prior = p.latest_assignment(current, line, f)
            if prior:
                links.append(f"`{current}` <- `{expr(prior[0])}`")
                evidence.extend(prior[1])
                return links, evidence, prior[0]
    return links, evidence, None


def transform_evidence(p, function):
    if not function:
        return [], []
    claims, evidence = [], []
    index = p.sources.index(function.lines[0])
    if index and p.sources[index - 1].label == function.lines[0].label and p.sources[index - 1].code.lstrip().startswith("@"):
        deco = p.sources[index - 1]
        claims.append(f"The saved function decorator is `{deco.code.strip()}`.")
        evidence.append(deco)
    for targets, value, refs in assignments(p.sources):
        if not isinstance(value, ast.Call):
            continue
        calls = [n for n in ast.walk(value) if isinstance(n, ast.Call) and callable_name(n.func) in {"grad", "jit", "make_jaxpr", "vmap"}]
        if any(any(isinstance(a, ast.Name) and a.id == function.name for a in c.args) for c in calls):
            claims.append("Saved source assigns `" + ", ".join(expr(t) for t in targets) + " = " + expr(value) + "`.")
            evidence.extend(refs)
    return claims, evidence


def scalar_conversion(p):
    error = p.error_line("ConcretizationTypeError")
    if not error:
        return None
    for frame in reversed(p.frames):
        for line in frame.lines:
            calls = [c for c in nodes(line.code, ast.Call)
                     if isinstance(c.func, ast.Name) and c.func.id in {"float", "int", "bool", "complex"} and c.args]
            if not calls or not line.arrow:
                continue
            call = calls[0]
            operand = call.args[0]
            f = p.function_for(frame, line)
            evidence = [line, error]
            claim = f"{call.func.id} receives `{expr(operand)}`"
            if isinstance(operand, ast.Name) and f:
                prior = p.latest_assignment(operand.id, line, f)
                if prior:
                    claim += f"; its nearest preceding local assignment is `{operand.id} = {expr(prior[0])}`"
                    evidence.extend(prior[1])
            if isinstance(operand, ast.Call):
                claim += f"; the conversion operand is the result of `{expr(operand.func)}`"
            trace_origin = next((x for x in p.trace_lines if "depends on" in x.code or "became a tracer" in x.code), None)
            if trace_origin:
                evidence.append(trace_origin)
            return result("scalar_operand", "trace_correlated_partial", claim + ".",
                          evidence, "The recorded error establishes tracedness at conversion. Saved source alone does not establish earlier runtime values or execution order; multiple conversions on the same recorded line are not individually disambiguated.")
    return None


def split_count(p):
    error = p.error_line("ConcretizationTypeError")
    if not error:
        return None
    for frame in p.frames:
        for line in frame.lines:
            for call in nodes(line.code, ast.Call):
                if callable_name(call.func) != "split" or len(call.args) < 2 or not line.arrow:
                    continue
                f = p.function_for(frame, line)
                count = call.args[1]
                if not f or not isinstance(count, ast.Name) or count.id not in f.args:
                    continue
                pos = f.args.index(count.id)
                evidence = [line, f.lines[0], error]
                claim = f"`{expr(call.func)}` receives `{count.id}` as its count; that value is formal argument {pos} of `{f.name}`"
                header_index = p.sources.index(f.lines[0])
                decorators = []
                for prev in reversed(p.sources[:header_index]):
                    if prev.label != f.lines[0].label or (prev.code.strip() and not prev.code.lstrip().startswith("@")):
                        break
                    if prev.code.lstrip().startswith("@"):
                        decorators.append(prev)
                for deco in decorators:
                    for dcall in nodes(deco.code.lstrip().lstrip("@"), ast.Call):
                        for kw in dcall.keywords:
                            if kw.arg == "static_argnums":
                                try:
                                    static = ast.literal_eval(kw.value)
                                    static = (static,) if isinstance(static, int) else tuple(static)
                                except (ValueError, TypeError):
                                    continue
                                if pos not in static:
                                    claim += f"; the shown decorator declares static positions {static}, excluding {pos}"
                                    evidence.append(deco)
                for tr in p.trace_lines:
                    if "int(num)" in tr.code:
                        claim += "; the recorded implementation passes this count to `int(num)`"
                        evidence.append(tr)
                        break
                return result("count_parameter", "trace_correlated_partial", claim + ".", evidence,
                              "The body and decorator establish the argument relationship. A type annotation does not itself make an argument static; no complete runtime argument history is available.")
    return None


def literal_value(p, node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value, []
    if isinstance(node, ast.Name):
        assignment = p.unique_global_assignment(node.id)
        if assignment and isinstance(assignment[0], ast.Constant):
            return assignment[0].value, assignment[1]
    return expr(node), []


def source_shape(p, node):
    if not isinstance(node, ast.Name):
        return None
    assignment = p.unique_global_assignment(node.id)
    if not assignment or not isinstance(assignment[0], ast.Call):
        return None
    call, lines = assignment
    shapes = [kw.value for kw in call.keywords if kw.arg == "shape"]
    if not shapes and callable_name(call.func) in {"zeros", "ones", "empty", "normal", "uniform"}:
        shapes = [call.args[0]] if call.args and callable_name(call.func) in {"zeros", "ones", "empty"} else [call.args[1]] if len(call.args) > 1 else []
    if len(shapes) != 1 or not isinstance(shapes[0], (ast.Tuple, ast.List)):
        return None
    values, evidence = [], list(lines)
    for dim in shapes[0].elts:
        value, refs = literal_value(p, dim)
        values.append(value)
        evidence.extend(refs)
    return values, evidence


def matmul_shape(p):
    error = next((x for x in p.trace_lines if "dot_general requires contracting dimensions" in x.code), None)
    if not error:
        return None
    for frame in reversed(p.frames):
        for line in frame.lines:
            products = nodes(line.code, ast.BinOp)
            products = [n for n in products if isinstance(n.op, ast.MatMult)]
            if not products or not line.arrow:
                continue
            f = p.function_for(frame, line)
            callers = p.callers(frame, f)
            if len(products) == 1:
                product = products[0]
                if isinstance(product.left, ast.Name) and isinstance(product.right, ast.Attribute) and product.right.attr == "T" and isinstance(product.right.value, ast.Name) and len(callers) == 1:
                    mapped, caller = callers[0]
                    left_name, right_name = product.left.id, product.right.value.id
                    ls = source_shape(p, mapped[left_name]) if left_name in mapped else None
                    rs = source_shape(p, mapped[right_name]) if right_name in mapped else None
                    if ls and rs and len(ls[0]) >= 1 and len(rs[0]) >= 2:
                        transposed = list(reversed(rs[0]))
                        claim = (f"Caller binds `{left_name}` to `{expr(mapped[left_name])}` with shown shape {ls[0]} and `{right_name}` to `{expr(mapped[right_name])}` with shown shape {rs[0]}. "
                                 f"In `{expr(product)}`, `.T` reverses all axes to {transposed}; the contracting axes are the left final axis ({ls[0][-1]}) and the transposed right penultimate axis ({transposed[-2]}).")
                        out = result("transpose_contraction", "syntactic_shape", claim,
                                      [line, caller, error] + ls[1] + rs[1],
                                      "Shapes are reconstructed from saved assignments, not a runtime replay; source order does not prove which array versions were live.")
                        reported = re.search(r"got\s+[\[(](\d+).*?and\s+[\[(](\d+)", error.code)
                        if reported and all(isinstance(x, int) for x in (ls[0][-1], transposed[-2])):
                            out["recorded_sizes_consistent"] = [ls[0][-1], transposed[-2]] == [int(reported[1]), int(reported[2])]
                            if not out["recorded_sizes_consistent"]:
                                out["kind"], out["tier"] = "shape_trace_conflict", "syntactic_conflict"
                                out["withheld_reason"] += " The reconstructed contracting sizes do not match the recorded error, so this source state is not promoted to the recorded cause."
                        return out
            evidence = [line, error]
            claim = "The recorded expression contains " + ", ".join("`" + expr(x) + "`" for x in products) + "."
            linked = False
            if len(callers) == 1:
                mapped, caller = callers[0]
                used = sorted({n.id for prod in products for n in ast.walk(prod) if isinstance(n, ast.Name)})
                links = [f"`{name}` is supplied as `{expr(mapped[name])}`" for name in used if name in mapped]
                if links:
                    claim += " Syntactic matching of the displayed call to the shown signature gives: " + "; ".join(links) + "."
                    evidence.append(caller)
                    linked = True
            if not linked:
                return result("abstained", "withheld", "", evidence,
                              "Only a matrix-product location was recovered; there is no corroborated cross-fact operand or axis relationship.")
            return result("contraction_call_binding", "trace_correlated_partial", claim, evidence,
                          "The traceback gives unequal contracting dimensions but does not assign them to every operand or distinguish multiple matrix products on this line. No dimension-to-operand mapping is asserted.")
    return None


def flatten_dot(p):
    error = next((x for x in p.trace_lines if "dot_general requires contracting dimensions" in x.code), None)
    if not error:
        return None
    for frame in reversed(p.frames):
        for line in frame.lines:
            for call in nodes(line.code, ast.Call):
                if callable_name(call.func) != "dot" or len(call.args) != 2 or not isinstance(call.args[1], ast.Name) or not line.arrow:
                    continue
                f = p.function_for(frame, line)
                if not f:
                    continue
                operand, current, evidence, links = call.args[1].id, call.args[1].id, [line, error], []
                visited = set()
                cursor = line
                while current not in visited:
                    visited.add(current)
                    prior = p.latest_assignment(current, cursor, f)
                    if not prior:
                        break
                    value, refs = prior
                    links.append(f"`{current} = {expr(value)}`")
                    evidence.extend(refs)
                    cursor = refs[0]
                    if isinstance(value, ast.Name):
                        current = value.id
                    else:
                        break
                if not links:
                    continue
                claim = f"The right dot operand `{operand}` follows the local chain " + " <- ".join(links) + "."
                if isinstance(value, ast.Call) and callable_name(value.func) == "reshape":
                    if isinstance(value.func, ast.Attribute):
                        receiver = expr(value.func.value)
                    elif value.args:
                        receiver = expr(value.args[0])
                    else:
                        receiver = ""
                    shape_bind = next(((targets, val, refs) for targets, val, refs in assignments(f.lines)
                                       if refs[0].physical < cursor.physical and isinstance(val, ast.Attribute)
                                       and val.attr == "shape" and expr(val.value) == receiver), None)
                    if shape_bind:
                        claim += " Its reshape factors are bound by `" + ", ".join(expr(t) for t in shape_bind[0]) + " = " + expr(shape_bind[1]) + "`."
                        evidence.extend(shape_bind[2])
                for s in p.sources:
                    for outer_call in nodes(s.code, ast.Call):
                        if callable_name(outer_call.func) == "vmap" and outer_call.args and isinstance(outer_call.args[0], ast.Name) and outer_call.args[0].id == f.name:
                            axes = next((k.value for k in outer_call.keywords if k.arg == "in_axes"), None)
                            if axes:
                                claim += f" The shown `vmap({f.name}, in_axes={expr(axes)})` determines which input axis is removed before this body runs."
                                evidence.append(s)
                return result("flatten_dot_axis", "syntactic_partial", claim, evidence,
                              "This establishes a local operand/axis relationship. It does not reconstruct the live weight shapes, input dimensions, initialization history, or assign reported sizes to operands.")
    return None


def index_carry(p):
    error = p.error_line("TracerArrayConversionError")
    if not error:
        return None
    for frame in reversed(p.frames):
        for line in frame.lines:
            f = p.function_for(frame, line)
            if not f or not line.arrow:
                continue
            for sub in nodes(line.code, ast.Subscript):
                indices = {n.id for n in ast.walk(sub.slice) if isinstance(n, ast.Name)}
                formal_indices = indices.intersection(f.args)
                if not formal_indices:
                    continue
                scans = [(call, s) for s in p.sources for call in nodes(s.code, ast.Call)
                         if callable_name(call.func) == "scan" and call.args and isinstance(call.args[0], ast.Name) and call.args[0].id == f.name]
                if not scans:
                    continue
                claim = f"`{expr(sub)}` uses callback argument(s) {sorted(formal_indices)} as index data."
                evidence = [line, f.lines[0], error]
                for scan, s in scans:
                    if len(scan.args) >= 2 and f.args:
                        claim += f" The displayed `{expr(scan.func)}` uses `{f.name}` as callback with carry formal `{f.args[0]}` and initial carry `{expr(scan.args[1])}`."
                        evidence.append(s)
                        break
                if isinstance(sub.value, ast.Name):
                    owner = p.latest_assignment(sub.value.id, line, f)
                    if owner:
                        claim += f" The local array alias is `{sub.value.id} = {expr(owner[0])}`."
                        evidence.extend(owner[1])
                    else:
                        for targets, val, refs in assignments(p.sources):
                            if isinstance(val, (ast.Tuple, ast.List)):
                                matches = [v for t in targets for name, v in bindings(t, val) if name == sub.value.id]
                                source_sites = [x for x in f.lines if x.code.strip() == line.code.strip()]
                                if matches and source_sites and refs[0].physical < source_sites[0].physical:
                                    claim += f" Saved unpacking binds `{sub.value.id}` from `{expr(matches[0])}`."
                                    evidence.extend(refs)
                                    break
                return result("scan_index_operand", "trace_correlated_partial", claim, evidence,
                              "The recorded error and callback establish the index/carry relationship. Saved aliases or annotations alone do not prove the live container type or its constructor history.")
    return None


def jit_wrapper(p):
    error = p.error_line("TracerArrayConversionError")
    if not error:
        return None
    for frame in p.frames:
        for line in frame.lines:
            for call in nodes(line.code, ast.Call):
                if not isinstance(call.func, ast.Name) or not line.arrow:
                    continue
                wrapper = p.unique_global_assignment(call.func.id)
                if not wrapper or not isinstance(wrapper[0], ast.Call) or callable_name(wrapper[0].func) != "jit" or not wrapper[0].args:
                    continue
                target = wrapper[0].args[0]
                evidence = [line, error] + wrapper[1]
                claim = f"The trace calls `{expr(call)}`; saved source assigns `{call.func.id}` to a JIT wrapper of `{expr(target)}`."
                if isinstance(target, ast.Name):
                    origin = p.unique_global_assignment(target.id)
                    if origin:
                        claim += f" The target is constructed by `{target.id} = {expr(origin[0])}`."
                        evidence.extend(origin[1])
                last = next((x for fr in reversed(p.frames) for x in reversed(fr.lines)
                             if x.arrow and not x.code.lstrip().startswith("raise ")), None)
                if last:
                    claim += f" The last recorded non-raise expression is `{last.code.strip()}`."
                    evidence.append(last)
                return result("jit_target_origin", "syntactic_partial", claim, evidence,
                              "This is a syntactic wrapper/constructor link. It does not prove saved binding identity at runtime, generated function globals, the identity of intermediary objects, or which nested subcall initiated conversion.")
    return None


def conversion_operand(p):
    error = p.error_line("TracerArrayConversionError")
    if not error:
        return None
    for frame in reversed(p.frames):
        for line in reversed(frame.lines):
            if not line.arrow or line.code.lstrip().startswith("raise "):
                continue
            calls = nodes(line.code, ast.Call)
            supported = {"array", "asarray", "power", "sin", "cos", "exp", "reduce_sum"}
            calls = [c for c in calls if callable_name(c.func) in supported and c.args]
            if not calls:
                continue
            call = calls[0]
            if len(calls) > 1:
                simple = [c for c in calls if all(not isinstance(a, (ast.Call, ast.List, ast.ListComp)) for a in c.args)]
                if simple:
                    call = simple[0]
            operand = call.args[0]
            f = p.function_for(frame, line)
            evidence = [line, error]
            claim = f"The recorded `{expr(call.func)}` call receives `{expr(operand)}` as its first operand."
            names = {n.id for n in ast.walk(operand) if isinstance(n, ast.Name)}
            if f:
                names.intersection_update(f.args)
            else:
                names.intersection_update(frame.args)
            if names:
                claim += " The operand depends on shown formal(s) " + ", ".join(f"`{n}`" for n in sorted(names)) + f" of `{frame.name}`."
                header = next((s for s in p.trace_lines if s.code == frame.label), None)
                if header:
                    evidence.append(header)
            for name in sorted(names):
                links, refs, origin = call_origin(p, frame, name)
                if links:
                    claim += " Syntactic call/signature matching gives " + "; ".join(links) + "."
                    evidence.extend(refs)
                    if isinstance(operand, ast.Attribute) and isinstance(operand.value, ast.Name) and operand.value.id == name and isinstance(origin, ast.Call):
                        value = next((k.value for k in origin.keywords if k.arg == operand.attr), None)
                        if value:
                            claim += f" That constructor binds selected field `{operand.attr}` from `{expr(value)}`."
            if f:
                for name in sorted(names):
                    prior = p.latest_assignment(name, line, f)
                    if prior:
                        claim += f" A preceding local assignment is `{name} = {expr(prior[0])}`."
                        evidence.extend(prior[1])
                evidence.append(f.lines[0])
                transforms, refs = transform_evidence(p, f)
                claim += (" " + " ".join(transforms)) if transforms else ""
                evidence.extend(refs)
            roots = {n.id for n in ast.walk(call.func) if isinstance(n, ast.Name)}
            imports = []
            for s in p.sources:
                for node in nodes(s.code, (ast.Import, ast.ImportFrom)):
                    for a in node.names:
                        bound = a.asname or (a.name.split(".")[0] if isinstance(node, ast.Import) else a.name)
                        if bound in roots:
                            imports.append((ast.unparse(node), s))
            unique = list(dict.fromkeys(x[0] for x in imports))
            if unique:
                claim += " Saved namespace evidence: " + "; ".join("`" + x + "`" for x in unique) + "."
                evidence.extend(x[1] for x in imports)
            for import_text, imported_line in imports:
                for imported in nodes(imported_line.code, ast.ImportFrom):
                    module_path = (imported.module or "").replace(".", "/") + ".py"
                    module_lines = [s for s in p.sources if s.label.replace("\\", "/").endswith(module_path)]
                    for alias in imported.names:
                        if (alias.asname or alias.name) not in roots:
                            continue
                        rebindings = []
                        for targets, value, refs in assignments(module_lines):
                            if any(isinstance(t, ast.Name) and t.id == alias.name for t in targets):
                                rebindings.append((value, refs))
                        global_decls = [s for s in module_lines for n in nodes(s.code, ast.Global) if alias.name in n.names]
                        if len(rebindings) > 1 and global_decls:
                            claim += (f" The source module rebinds `{alias.name}` to " + ", ".join(f"`{expr(v)}`" for v, _ in rebindings) +
                                      f"; a direct `from ... import {alias.name}` alias retains the imported object when the module name is later rebound.")
                            evidence.extend(s for _, refs in rebindings for s in refs)
                            evidence.extend(global_decls)
            return result("conversion_operand_call", "trace_correlated_partial", claim, evidence,
                          "All call/signature and namespace links are syntactic. Filtered frames can hide intermediary calls; matching source names is not runtime identity proof. The live backend object, nested subcall responsible for conversion, and values inside an operand remain unresolved unless separately recorded.")
    return None


def analyze(text):
    p = Packet(text)
    for rule in (split_count, scalar_conversion, matmul_shape, flatten_dot, index_carry, jit_wrapper, conversion_operand):
        found = rule(p)
        if found:
            found["rule"] = rule.__name__
            if any("locally authored pseudocode" in s.label for s in p.sources):
                found["evidence_basis"] = "archived_factual_trace_plus_locally_authored_pseudocode"
            else:
                found["evidence_basis"] = "recorded_trace_plus_saved_source" if p.sources else "recorded_trace_only"
            found["relationship_scope"] = "Syntactic reconstruction tied to recorded expressions; dynamic evidence is limited to the supplied traceback, not a replay by this analyzer."
            found["complete_diagnosis"] = False
            return found
    return result("abstained", "withheld", "", [], "No implemented rule found a corroborated operand, caller, or axis relationship.")
