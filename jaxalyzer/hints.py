"""Propose source relationships from a flat packet. Packet code is never executed."""
import ast
import math
import re

from . import base

HEADER = "\n\nDERIVED RELATIONSHIPS\n"
ERROR = re.compile(r"^(?:[\w.]+\.)?(\w*(?:Error|Exception)):\s*(.*)$")
DIM = r"(?:\([0-9,\s-]*\)|\[[0-9,\s-]*\]|-?\d+)"
CTOR = {"np.dot", "jnp.dot", "numpy.dot", "jax.numpy.dot", "np.matmul", "jnp.matmul", "numpy.matmul", "jax.numpy.matmul"}
SHAPE_MODULE = re.compile(r"(?:^|/)jax/(?:_src/)?numpy/")
DEPENDENCY = re.compile(r"(?:The error occurred while tracing|While tracing) the function ([\w.<>]+) at .+? for ([\w]+)[.,] [Tt]his concrete value was not available in Python because it depends on the value of the argument ['`]?([\w.]+)['`]?\.$")


def ordinal(i):
    names = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth")
    if i <= len(names):
        return names[i-1]
    suffix = "th" if 10 < i % 100 < 14 else {1:"st", 2:"nd", 3:"rd"}.get(i % 10, "th")
    return f"{i}{suffix}"


def render_previous(text, prior, raw):
    """Render the actual previously admitted hint, keeping any existing qualifications."""
    if not prior:
        return None
    if raw["kind"] == "count_parameter":
        prior = re.sub(r"formal argument (\d+) of", lambda m: f"the {ordinal(int(m[1])+1)} positional parameter (zero-based index {m[1]}) of", prior)
    limits = raw.get("withheld_reason", "")
    if raw["kind"] == "reviewed_hash_association":
        limits = "This association is supported for this recorded failure and source snapshot."
    if raw["kind"] == "jit_target_origin":
        limits = "The wrapper and constructor links come from saved source. Execution history and live namespace bindings are only established where the trace shows them."
        packet = base.Packet(text)
        last = next((l for f in reversed(packet.frames) for l in reversed(marked(f)) if not l.code.lstrip().startswith("raise ")),None)
        if last and len(base.nodes(last.code,ast.Call))>1:
            limits += " The trace does not identify which nested call first requested conversion."
    if not limits:
        raise ValueError("A delivered relationship requires an evidence limit")
    if raw["kind"] == "conversion_operand_call":
        packet = base.Packet(text)
        for i,frame in enumerate(packet.frames):
            if frame.name != "closure_convert" or "/jax/" not in frame.label.replace("\\","/") or i==0:
                continue
            caller = packet.frames[i-1]
            sites = [line for line in caller.lines if line.arrow and any(base.expr(n.func).endswith(".closure_convert") for n in base.nodes(line.code,ast.Call))]
            if len(sites)==1:
                prior += f" The recorded call path enters JAX closure conversion at `{sites[0].code.strip()}`."
                limits += " Closure conversion traces the callback; it does not establish that a derivative was being computed."
                raw = dict(raw, evidence=raw.get("evidence",[])+[base.cite(sites[0])])
                break
    return dict(raw,relationship=prior,uncertainty=limits,hint=prior+"\n"+limits)


def marked(frame):
    return [l for l in frame.lines if l.arrow and not l.code.lstrip().startswith("[...") and l.code.strip()]


def terminal(packet):
    hits=[(l,ERROR.fullmatch(l.code)) for l in packet.trace_lines if ERROR.fullmatch(l.code)]
    if len(hits)!=1:
        return None
    line,match=hits[0]
    trace=[l.code for l in packet.trace_lines]
    if any("During handling of the above exception" in s or "direct cause of the following exception" in s for s in trace):
        return None
    tail=[l.code.strip() for l in packet.trace_lines if l.physical>line.physical and l.code.strip()]
    if any(s.startswith("Traceback (") or re.fullmatch(r"(?:\w+\.)*\w*(?:Error|Exception)",s) for s in tail):
        return None
    if match[1]=="TypeError" and any(not s.startswith(("--------------------","For simplicity, JAX has removed its internal frames")) for s in tail):
        return None
    return line,match[1],match[2]


def make(kind,claim,limit,refs):
    result=base.result(kind,"trace_correlated_partial",claim,refs,limit)
    return dict(result,uncertainty=limit,hint=claim+"\n"+limit,complete_diagnosis=False)


def shape_site(p, name):
    # Stop at the first application frame above the matching JAX implementation.
    impls=[i for i,f in enumerate(p.frames) if SHAPE_MODULE.search(f.label.replace("\\","/")) and f.name.rsplit(".",1)[-1]==name]
    if len(impls)!=1:
        return None
    i=impls[0]
    if any("/jax/" not in f.label.replace("\\","/") for f in p.frames[i+1:]):
        return None
    inner=marked(p.frames[i])
    if not inner:return None
    if name in {"dot","matmul"} and not any(re.search(r"\b(?:lax\.)?dot_general\(",l.code) for l in inner):
        return None
    for f in reversed(p.frames[:i]):
        if SHAPE_MODULE.search(f.label.replace("\\","/")) and "/array_methods.py" in f.label.replace("\\","/"):
            continue
        active=marked(f)
        if len(active)==1:
            return f,active[0],inner[0]
        break
    return None


def preceding(frame,line,expr):
    if not isinstance(expr,ast.Name):
        return "", []
    for l in frame.lines:
        if l.number is None or l.number+1!=line.number or l.physical>=line.physical:
            continue
        parsed=base.tree(l.code)
        if parsed and len(parsed.body)==1 and isinstance(parsed.body[0],ast.Assign):
            a=parsed.body[0]
            if len(a.targets)==1 and isinstance(a.targets[0],ast.Name) and a.targets[0].id==expr.id:
                return f" The preceding displayed line assigns `{expr.id} = {base.expr(a.value)}`.", [l]
    return "", []


def dot(p,error,message):
    found=re.fullmatch(r"dot_general requires contracting dimensions to have the same shape, got [\[(](\d+),?[\])] and [\[(](\d+),?[\])][.]?",message)
    if not found or found[1]==found[2]:
        return None
    for method in ("dot","matmul"):
        site=shape_site(p,method)
        if not site:
            continue
        frame,line,implementation=site
        tree=base.tree(line.code)
        if not tree or len(tree.body)!=1 or implementation.physical>=error.physical or any(isinstance(n,(ast.IfExp,ast.BoolOp,ast.comprehension,ast.Lambda)) for n in ast.walk(tree)):
            continue
        products=[n for n in ast.walk(tree) if (isinstance(n,ast.BinOp) and isinstance(n.op,ast.MatMult)) or
                  (isinstance(n,ast.Call) and base.callable_name(n.func) in {"dot","matmul","tensordot","einsum","dot_general"})]
        if len(products)!=1:
            continue
        op=products[0]
        if isinstance(op,ast.BinOp) and method=="matmul":
            left,right=op.left,op.right
        elif isinstance(op,ast.Call) and base.expr(op.func) in CTOR and base.callable_name(op.func)==method and len(op.args)>=2 and not any(isinstance(n,ast.Starred) for n in op.args):
            left,right=op.args[:2]
        else:
            continue
        l,r=base.expr(left),base.expr(right)
        ordered=bool(re.search(r"\bdot_general\(\s*a\s*,\s*b\s*,",implementation.code)) and method=="dot"
        claim=f"The marked `{base.expr(op)}` site is followed by JAX's `{method}` frame. For non-scalar inputs, its contraction compares the final axis of `{l}` with the penultimate axis of `{r}` (its sole axis for a vector)."
        if ordered:
            claim+=f" The displayed call forwards `a, b` in order; the shape error reports left extent {found[1]} and right extent {found[2]}."
        else:
            claim+=f" The underlying shape check reports contracting extents {found[1]} and {found[2]}."
        extra,refs=preceding(frame,line,right)
        limit="The reported extents describe contracted axes. They do not establish either operand's full shape or how that shape was produced."
        return make("contraction_axes",claim+extra,limit,[line,implementation,error]+refs)


def dims(s):
    v=ast.literal_eval(s)
    return (v,) if type(v) is int else tuple(v)


def reshape(p,error,message):
    m=re.fullmatch(r"cannot reshape array of shape ("+DIM+r") \(size (\d+)\) into shape ("+DIM+r") (?:\(size (\d+)\)|because the product of specified axis sizes \((\d+)\) does not evenly divide (\d+))[.]?",message)
    if not m:
        return None
    site=shape_site(p,"_compute_newshape")
    if not site:
        return None
    frame,line,implementation=site
    tree=base.tree(line.code)
    if not tree or len(tree.body)!=1 or any(isinstance(n,(ast.IfExp,ast.BoolOp,ast.comprehension)) for n in ast.walk(tree)):
        return None
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="reshape"]
    if len(calls)!=1 or implementation.physical>=error.physical:
        return None
    call=calls[0]
    if not call.args or call.keywords or base.expr(call.func.value) in {"np","jnp","numpy","jax.numpy"}:
        return None
    try:
        src,dest=dims(m[1]),dims(m[3])
    except (SyntaxError,ValueError,TypeError):
        return None
    if not src or not dest or any(type(v) is not int or v<0 for v in src) or any(type(v) is not int or v< -1 for v in dest) or dest.count(-1)>1 or math.prod(src)!=int(m[2]):
        return None
    fixed_product=math.prod(n for n in dest if n!=-1)
    if m[4]:
        if -1 in dest or fixed_product!=int(m[4]) or fixed_product==int(m[2]):return None
    elif -1 not in dest or not fixed_product or fixed_product!=int(m[5]) or int(m[6])!=int(m[2]) or int(m[2])%fixed_product==0:return None
    standalone_shape=len(call.args)==1 and not isinstance(call.args[0],(ast.Tuple,ast.List))
    targets=call.args
    if len(targets)==1 and isinstance(targets[0],(ast.Tuple,ast.List)):targets=targets[0].elts
    starred=[i for i,n in enumerate(targets) if isinstance(n,ast.Starred)]
    if starred and starred!=[len(targets)-1]:return None
    fixed=targets[:starred[0]] if starred else targets
    if (not starred and len(fixed)!=len(dest)) or len(fixed)>len(dest):
        return None
    slots=[]
    for n,v in zip(fixed,dest):
        if isinstance(n,(ast.Constant,ast.UnaryOp)):
            try:
                if ast.literal_eval(n)!=v:return None
            except (ValueError,TypeError):return None
        else:
            if standalone_shape:
                slots.append(f"the target expression `{base.expr(n)}` supplies a single axis of length {v}")
            else:
                slots.append(f"`{base.expr(n)}` = {v}")
    if not slots:return None
    extra,refs=preceding(frame,line,call.func.value)
    claim=f"The marked reshape of `{base.expr(call.func.value)}` requests these target dimensions in order: "+", ".join(slots)+". "
    claim+=f"The recorded operand has {int(m[2])} elements; "
    claim+=(f"the target needs {fixed_product}." if m[4] else f"the specified axes have product {fixed_product}, which cannot divide the operand size to infer the remaining axis.")
    limit="This maps the recorded target sizes to the displayed reshape. Earlier input shapes and the reason those sizes were chosen require separate evidence."
    return make("reshape_target_slots",claim+extra,limit,[line,implementation,error]+refs)


def dependency(p,error):
    candidates=[]
    for record in p.trace_lines:
        if record.physical<=error.physical:continue
        between=[l.code for l in p.trace_lines if error.physical<l.physical<record.physical]
        if any(s.strip() and not s.startswith("The problem arose with the ") for s in between):continue
        match=DEPENDENCY.fullmatch(record.code)
        if not match:continue
        func,transform,arg=match.groups()
        if not all(s.isidentifier() for s in arg.split(".")):continue
        frames=[f for f in p.frames if f.name.split(".")[-1]==func.split(".")[-1]]
        if len(frames)!=1:continue
        frame=frames[0]
        active=marked(frame)
        if len(active)!=1 or active[0].physical>=error.physical:continue
        line=active[0]; parsed=base.tree(line.code)
        if not parsed or len(parsed.body)!=1 or any(isinstance(n,(ast.comprehension,ast.Lambda)) for n in ast.walk(parsed)):continue
        contexts=[n for n in ast.walk(parsed) if isinstance(n,(ast.Call,ast.BinOp,ast.Subscript,ast.Compare,ast.Attribute)) and any(isinstance(v,(ast.Name,ast.Attribute)) and base.expr(v)==arg and isinstance(v.ctx,ast.Load) for v in ast.walk(n))]
        if not contexts:continue
        # Name the narrowest visible use, without claiming it caused conversion.
        ctx=min(contexts,key=lambda n:len(list(ast.walk(n))))
        claim=f"JAX reports a dependency on `{arg}` while tracing `{func}` for `{transform}`. The marked frame for that function uses it in `{base.expr(ctx)}`."
        limit="This identifies a reported dependency and its source use. It does not identify a particular nested operation as the first conversion or establish a concrete argument value."
        candidates.append(make("reported_argument_use",claim,limit,[line,error,record]))
    return candidates[0] if len(candidates)==1 else None


def caller_operand(p,error):
    """Join an explicit argument or method receiver across adjacent displayed frames."""
    for idx in range(len(p.frames)-1,0,-1):
        caller,callee=p.frames[idx-1:idx+1]
        if any("/jax/" not in f.label.replace("\\","/") or f.name.rsplit(".",1)[-1] not in {"__int__","__float__","__array__","error"} for f in p.frames[idx+1:]):
            continue
        name=callee.name.rsplit(".",1)[-1]
        called,inside=marked(caller),marked(callee)
        if not name.isidentifier() or name.startswith("__") or len(called)!=1 or len(inside)!=1 or not callee.args:
            continue
        a,b=called[0],inside[0]
        if not a.physical<b.physical<error.physical:continue
        conversions=[n for n in base.nodes(b.code,ast.Call) if base.callable_name(n.func) in {"float","int","array","asarray","asanyarray"} and n.args]
        sites=[n for n in base.nodes(a.code,ast.Call) if base.callable_name(n.func)==name]
        if len(conversions)!=1 or len(sites)!=1:continue
        conv,site=conversions[0],sites[0]
        if not isinstance(conv.args[0],ast.Name) or any(isinstance(n,ast.Starred) for n in site.args) or any(k.arg is None for k in site.keywords):continue
        formal=conv.args[0].id
        if formal not in callee.args:continue
        if callee.args[0] in {"self","cls"}:
            if formal!="self" or not isinstance(site.func,ast.Attribute) or site.args or site.keywords:continue
            actual=site.func.value
            claim=f"The caller invokes `{base.expr(site)}`. The following `{name}` frame uses `self` in `{base.expr(conv)}`. Under ordinary instance-method binding its receiver is `{base.expr(actual)}`."
        else:
            i=callee.args.index(formal)
            values=([site.args[i]] if i<len(site.args) else [])+[k.value for k in site.keywords if k.arg==formal]
            if len(values)!=1:continue
            actual=values[0]
            claim=f"The marked call `{base.expr(site)}` supplies `{base.expr(actual)}` for `{formal}` under the displayed signature of `{name}`. Its frame uses `{formal}` in `{base.expr(conv)}`."
        return make("conversion_frame_binding",claim,"This matches a displayed call and conversion expression. Filtered frames and earlier assignments can leave the runtime object and its origin unresolved.",[a,b,error])


def expansion(text):
    p=base.Packet(text);end=terminal(p)
    if not end:return None
    e,typ,msg=end
    if typ=="TypeError":
        return dot(p,e,msg) or reshape(p,e,msg)
    if typ in {"TracerArrayConversionError","ConcretizationTypeError"}:
        return dependency(p,e) or caller_operand(p,e)
    return None
