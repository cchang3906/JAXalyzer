"""Compute diagnostic evidence from the supplied traceback and archived source only."""
import ast
from collections import defaultdict
from dataclasses import dataclass
import math
import re

from . import extended as previous
Trace=previous.Trace
expr=ast.unparse

@dataclass(frozen=True)
class Source:
    number:int
    context:str
    code:str

def source_rows(text):
    section=False;ctx="source";rows=[]
    for i,raw in enumerate(text.splitlines(),1):
        s=raw.strip()
        if s.startswith(("SOURCE CONTEXT","SOURCE EXCERPTS","SUPPLEMENTAL RECORDED TRACEBACK")):
            section=True;ctx="source";continue
        if s.startswith(("SOURCE ATTRIBUTION","DERIVED RELATIONSHIPS")):section=False
        if not section or not s:continue
        if re.match(r'^(?:Notebook cell|Cell \d+|Excerpt \d+|Archived .+ lines|/?[\w./-]+\.py\s*$)',s,re.I):ctx=s[:110];continue
        if s.startswith(("Source:","License:","Author:","Attribution:","Git blob:","Failing cell:","Evidence representation","Owner group:","Source stratum:", "Pinned source:","Original factual reconstruction")):continue
        if s.startswith(("AUTHOR UNCERTAINTY","EVIDENCE LIMITS")):ctx="source limits";continue
        if s.startswith("No runtime values beyond the saved source"):continue
        if re.match(r'(?:\d+|L\d+(?:[\dL,\- ]*))(?:\s*[:,]\s*|\s{1,})',s) or ctx=="source limits" or s.lower().startswith(("uncertainty:","limit:")) or re.search(r'=|\b(?:def|class|return|import|from) |\w+\(',s):
            if len(s)<=650:rows.append(Source(i,ctx,s))
    return rows

def code_of(row):
    return re.sub(r'^(?:\d+(?::\s*|\s+)|L[\dL,\- ]+[:,]\s*(?:(?:factual|archived)[^:]{0,35}:\s*)?)','',row.code).strip()

def tree(s):
    try:return ast.parse(s)
    except (ValueError,SyntaxError):return None

STOP=set("if for else and or in is self cls def return the a an of from import as shown shape numpy jax jnp np at trace method source axis call marked recorded under that asarray array dot matmul reshape data type class shape".split())
def ids(s):
    return {n for n in re.findall(r'(?<!\w)[A-Za-z_]\w*',s) if n not in STOP}

def assigned(code):
    t=tree(code)
    if not t:return set()
    result=set()
    for n in t.body:
        if isinstance(n,(ast.Assign,ast.AnnAssign)):
            for target in (n.targets if isinstance(n,ast.Assign) else [n.target]):
                result.update(x.id for x in ast.walk(target) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Store))
        elif isinstance(n,(ast.FunctionDef,ast.ClassDef)):result.add(n.name)
    return result

def provenance(rows,seeds,frames):
    """Retrieve named producers plus their upstream source records, with exact packet offsets."""
    names=set(seeds); selected={}; scored=[]
    for row in rows:
        code=code_of(row)
        if not code or code.startswith(('#','http','"Work"')):continue
        words=ids(code);defs=assigned(code)
        if not defs:
            defs=set(re.findall(r'(?<![\w.])([A-Za-z_]\w*)\s*(?<![<>=!])=(?!=)',code))
        importance=2*int(bool(re.search(r'\b(?:vmap|pmap|scan|grad|jit|split|sample|reshape|transpose|stack|column_stack|arange|Linear|Dense|NUTS|MCMC)\b',code)))
        scored.append((row,words,defs,importance))
    # Exact assignment links are followed preferentially; prose links stay verbatim.
    for depth in range(3):
        nxt=set()
        for row,words,defs,imp in scored:
            if row.number in selected or not (defs & names):continue
            selected[row.number]=(13-depth*2+min(3,len(words&seeds))+imp,row)
            nxt.update(words-defs)
        names=nxt-names
        if not names:break
    trace_names={k for f in frames for k in ids(f.name) if len(k)>2}
    for row,words,defs,imp in scored:
        code=code_of(row)
        overlap=words&seeds
        strength=2*len({k for k in overlap if len(k)>1})+2*int(bool(overlap))+3*int(bool(words&trace_names))+imp
        if row.context=="source limits" or row.code.lower().startswith(("uncertainty:","limit:")):strength+=10
        if re.search(r'\b(?:shape|size|dimensions?|widths?|steps?|rows|columns|initial|construct|generat\w*|cast|pad|split|stack\w*|reshape\w*)\b',code,re.I) and ('factual' in row.code or overlap):strength+=3
        if row.number not in selected and strength>=3:selected[row.number]=(strength,row)
    best=[r for _,r in sorted(selected.values(),key=lambda x:(-x[0],x[1].number))[:10]]
    constants=set().union(*(set(re.findall(r'\b[A-Z][A-Z_0-9]{2,}\b',code_of(r))) for r in best))
    extra=[r for r,_,defs,_ in scored if defs&constants and r not in best]
    return sorted(best+extra[:3],key=lambda r:r.number)

def literal_shape(n):
    if isinstance(n,ast.Constant) and type(n.value) in (int,float,complex,bool):return ()
    if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.USub,ast.UAdd)):return literal_shape(n.operand)
    if not isinstance(n,(ast.Tuple,ast.List)):return None
    dims=[literal_shape(z) for z in n.elts]
    if not dims:return (0,)
    if dims[0] is None or any(x!=dims[0] for x in dims):return None
    return (len(dims),)+dims[0]

def shape_of(code,env):
    try:n=ast.parse(code,mode="eval").body
    except (ValueError,SyntaxError):return None
    if isinstance(n,ast.Name):return env.get(n.id)
    if isinstance(n,ast.Attribute):
        if n.attr=="T":
            sh=shape_of(expr(n.value),env);return None if sh is None else sh[::-1]
        return env.get(expr(n))
    if isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Constant) and n.slice.value is None:
        sh=shape_of(expr(n.value),env);return None if sh is None else (1,*sh)
    if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):
        op=n.func.attr
        if expr(n.func).split('.')[0] not in ("np","numpy","jnp","jax","jr"):return None
        if op in ("array","asarray") and n.args:return literal_shape(n.args[0])
        if op in ("ones","zeros","normal"):
            arg=next((v.value for v in n.keywords if v.arg=="shape"),None)
            if arg is None and op in ("ones","zeros") and n.args:arg=n.args[0]
            if arg is not None:
                try:sh=ast.literal_eval(arg)
                except (ValueError,SyntaxError):return None
                if type(sh) is int:sh=(sh,)
                if isinstance(sh,(list,tuple)) and all(type(x) is int and x>=0 for x in sh):return tuple(sh)
        if op=="arange" and len(n.args)==1 and not n.keywords and isinstance(n.args[0],ast.Constant) and type(n.args[0].value) is int and n.args[0].value>=0:
            return (n.args[0].value,)
    return None

def source_shapes(rows):
    shapes=defaultdict(set);witness=defaultdict(list)
    for r in rows:
        code=code_of(r);t=tree(code);sh=None;name=None
        if t and len(t.body)==1 and isinstance(t.body[0],ast.Assign):
            a=t.body[0]
            if len(a.targets)==1 and isinstance(a.targets[0],ast.Name):
                name=a.targets[0].id;sh=shape_of(expr(a.value),{})
        m=re.search(r'\b([A-Za-z_]\w*) is (?:a|an|generated by) [^;]{0,60}?\bshape (\([\d ,]+\))',code)
        if m:
            val=ast.literal_eval(m[2])
            if isinstance(val,tuple):name=m[1];sh=val
        symbolic=re.search(r'\b(\w+) is (?:a|an) [^;]{0,55}?array of shape (\([\w., ]+\))',code)
        if symbolic:
            parts=[x.strip() for x in symbolic[2][1:-1].split(',') if x.strip()]
            if parts and all(x.isdigit() or re.fullmatch(r'[A-Za-z_]\w*(?:\.\w+)*',x) for x in parts):
                name=symbolic[1];sh=tuple(int(x) if x.isdigit() else x for x in parts)
        m=re.search(r'\b(\w+) is (?:a|an) [^.;]{0,45}array with rows (\[[\d., \-]+\]) and (\[[\d., \-]+\])',code)
        if m:
            name=m[1];sh=literal_shape(ast.parse('['+m[2]+','+m[3]+']',mode='eval').body)
        if name and sh is not None:shapes[name].add(sh);witness[name].append(r)
    return {k:next(iter(v)) for k,v in shapes.items() if len(v)==1},witness

def dimension_facts(code,extents,shapes):
    tr=tree(code)
    if tr is None or any(isinstance(n,(ast.IfExp,ast.Lambda,ast.BoolOp,ast.comprehension)) for n in ast.walk(tr)):return ""
    ops=[n for n in ast.walk(tr) if isinstance(n,ast.BinOp) and isinstance(n.op,ast.MatMult) or isinstance(n,ast.Call) and previous.leaf(n.func) in ('dot','matmul','tensordot')]
    if len(ops)!=1:return ""
    n=ops[0]
    if isinstance(n,ast.BinOp):left,right=n.left,n.right;op='matmul'
    else:
        if len(n.args)<2:return ""
        left,right=n.args[:2];op=previous.leaf(n.func)
    axes=None
    if op=='tensordot':
        node=n.args[2] if len(n.args)>2 else next((k.value for k in n.keywords if k.arg=="axes"),None)
        try:axes=ast.literal_eval(node)
        except (TypeError,ValueError,SyntaxError):return ""
        if not isinstance(axes,(list,tuple)) or len(axes)!=2 or not all(type(i) is int for i in axes):return ""
    parts=[]
    for side,arg,num,extent in zip(("left","right"),(left,right),(0,1),extents):
        nm=expr(arg);sh=shape_of(nm,shapes);axis=axes[num] if axes else (-1 if num==0 else -2)
        if sh and not axes and num==1 and len(sh)==1:axis=0
        if sh and -len(sh)<=axis<len(sh) and sh[axis]==extent:
            axis=axis%len(sh)
            s=f"{side} `{nm}`: axis {axis} = {extent}. Source shape {sh} is consistent with this observed extent."
            if isinstance(arg,ast.Subscript) and isinstance(arg.slice,ast.Constant) and arg.slice.value is None:s+=" The inserted singleton is axis 0."
        else:
            label=(f"axis {axis}" if axes else "last axis (matrix columns)" if num==0 else "penultimate axis for rank ≥2 (matrix rows), axis 0 for a vector")
            s=f"{side} `{nm}`: {label} = {extent}."
        parts.append(s)
    return " ".join(parts)

def position_facts(t):
    result=[]
    for f,l in list(previous.focal(t))[:4]:
        n=previous.selected(f,l)
        if n is None:continue
        body=tree(l.code.strip())
        if body is None:continue
        matching=[a for a in ast.walk(body) if type(a)==type(n) and previous.pos(a)==previous.pos(n)]
        if len(matching)!=1:continue
        node=matching[0];anc={child:root for root in ast.walk(body) for child in ast.iter_child_nodes(root)}
        path=node;outer=None
        while path in anc:
            path=anc[path]
            if isinstance(path,(ast.Lambda,ast.GeneratorExp,ast.ListComp,ast.SetComp,ast.DictComp)):break
            if isinstance(path,ast.Call):outer=path;break
        if outer is not None:
            result.append(f"Packet line {l.physical}, `{f.name}`: caret selects `{expr(node)}` inside `{expr(outer.func)}(...)`. Python has to evaluate that subexpression before invoking the enclosing call.")
        elif isinstance(node,ast.Call) and f==next(previous.focal(t))[0]:
            result.append(f"Deepest marked non-JAX frame (packet line {l.physical}, `{f.name}`): the caret selects `{expr(node)}`."+
                          (" Its positional input expressions are "+", ".join(f"`{expr(arg)}`" for arg in node.args)+"." if node.args else ""))
    return result

def shape_evidence(t,reviewed,source,shapes):
    facts=[];match=previous.SHAPE.fullmatch(t.message)
    if match:
        for nm,raw in re.findall(r'`([\w.]+)` to `[^\x60]+` with shown shape (\[[\d ,]+\])',reviewed):
            sh=ast.literal_eval(raw)
            if all(type(k) is int and k>=0 for k in sh):shapes[nm]=tuple(sh)
        for f,l in previous.focal(t):
            if not any(z in reviewed for z in ('contraction','contracting','contract')):break
            allowed=re.findall(r'`([^`]+)`',reviewed.splitlines()[0])
            node=tree(l.code.strip())
            ops=[] if not node else [n for n in ast.walk(node) if isinstance(n,ast.BinOp) and isinstance(n.op,ast.MatMult) or isinstance(n,ast.Call) and previous.leaf(n.func) in ('dot','matmul','tensordot')]
            if len(ops)!=1:continue
            op=ops[0];term=expr(op)
            if not any(s.replace(" ","")==term.replace(" ","") for s in allowed):continue
            if 'does not assign' in reviewed or 'does not mark an individual' in reviewed:continue
            # Withhold the extent assignment if another non-JAX application runs below this frame.
            if any(not re.search(r'/(?:site|dist)-packages/(?:jax|jaxlib)/|/jax/_src/',previous.normpath(fr.header.code)) for fr in t.frames[t.frames.index(f)+1:]):continue
            vals=(int(match[1]),int(match[2]))
            mapped=dimension_facts(l.code.strip(),vals,shapes)
            lowering=" ".join(z.code.strip() for ff in t.frames[t.frames.index(f)+1:] if ff.name in ('dot','matmul','tensordot') for z in ff.lines)
            forwarded=bool(re.search(r'\bdot_general\(\s*a\s*,\s*b\s*,',lowering))
            if re.search(r'\bdot_general\(\s*b\s*,\s*a\s*,',lowering):return []
            conditional="" if forwarded or "Source shape" in mapped or "left extent" in reviewed else "Under an order-preserving lowering of this operation, "
            if mapped:facts.append(f"Contraction at packet line {l.physical}, `{term}`. {conditional}{mapped}")
            stacklines=[r for r in source if re.search(r'column[-_]stack',r.code,re.I)]
            if mapped and stacklines:
                facts.append(f"The source explicitly describes column stacking (packet line {stacklines[-1].number}). Column-stacking equal-length 1D vectors makes a matrix with one column per vector and one row per element. At `W @ X` the right matrix contracts its row axis, including when it came from that stack.")
            break
        return facts
    if t.message.startswith('cannot reshape'):
        dims=re.match(r'cannot reshape array of shape (\([\d ,]*\)) \(size (\d+)\) into shape (\([\d ,\-]*\))',t.message)
        if not dims:return []
        before=ast.literal_eval(dims[1]);after=ast.literal_eval(dims[3])
        if not isinstance(before,tuple) or not isinstance(after,tuple) or math.prod(before)!=int(dims[2]):return []
        facts.append(f"Reshape evidence from packet line {t.error.physical}: the object RECEIVING reshape has shape {before} ({math.prod(before)} elements); the REQUESTED target is {after}.")
        for f,l in previous.focal(t):
            ops=[n for n in previous.calls(l.code) if isinstance(n.func,ast.Attribute) and n.func.attr=="reshape"]
            if len(ops)!=1:continue
            op=ops[0]
            if expr(op.func.value) in ('np','jnp','jax.numpy','numpy'):continue
            args=op.args if len(op.args)>1 else op.args[0].elts if op.args and isinstance(op.args[0],(ast.Tuple,ast.List)) else op.args
            facts.append(f"The receiver in the displayed call is `{expr(op.func.value)}`; it supplies the input elements. The target expression is `{', '.join(map(expr,op.args))}`.")
            if isinstance(op.func.value,ast.Subscript) and isinstance(op.func.value.slice,ast.Slice) and math.prod(before)==0 and -1 not in after and math.prod(after)>0:
                facts.append(f"The empty slice has zero elements whereas the target needs {math.prod(after)}. A clipped or out-of-range slice can be empty even when its written bounds are different; the recorded sizes do not imply equal offsets or an empty target.")
            if len(args)==len(after) and not any(isinstance(a,ast.Starred) for a in args):
                for r in source:
                    tr=tree(code_of(r))
                    if not tr or len(tr.body)!=1 or not isinstance(tr.body[0],ast.Assign):continue
                    a=tr.body[0]
                    if len(a.targets)!=1 or not isinstance(a.targets[0],ast.Tuple) or not isinstance(a.value,ast.Attribute) or a.value.attr!='shape':continue
                    names=[expr(x) for x in a.targets[0].elts]
                    for i,v in enumerate(args):
                        if isinstance(v,ast.Name) and names.count(v.id)==1:
                            facts.append(f"The shown assignment at packet line {r.number} binds `{v.id}` from axis {names.index(v.id)} of `{expr(a.value.value)}`. The observed target slot {i} is {after[i]}.")
            break
    return facts

def binding_facts(source,marked):
    out=[]; imports=defaultdict(list);defs=defaultdict(list);parsed=[]
    for r in source:
        code=code_of(r);tr=tree(code)
        if tr and len(tr.body)==1:
            n=tr.body[0];parsed.append((r,n))
            if isinstance(n,(ast.Import,ast.ImportFrom)):
                for a in n.names:
                    path=(n.module+"." if isinstance(n,ast.ImportFrom) and n.module else "")+a.name
                    if isinstance(n,ast.Import) and a.asname is None:path=path.split('.')[0]
                    imports[a.asname or a.name.split('.')[0]].append((r,path,a.name,isinstance(n,ast.ImportFrom)))
        if code.startswith('def ') and code.rstrip().endswith(':'):
            d=tree(code+'\n    pass')
            if d and isinstance(d.body[0],ast.FunctionDef):defs[d.body[0].name].append((r,d.body[0]))
    for name,occ in imports.items():
        if not any(re.search(r'\b'+re.escape(name)+r'\.',l.code) for f,l in marked):continue
        distinct=list(dict.fromkeys(p for r,p,_,_ in occ))
        if len(distinct)>1:
            out.append("Source import order for `"+name+"`: "+" → ".join(f"`{p}` (packet line {r.number})" for r,p,_,_ in occ[:6])+". A later executed import in the same namespace replaces the earlier alias. Numbered notebook cells alone cannot certify execution order.")
        for r,p,orig,from_imp in occ:
            rebindings=[(z,expr(n.value)) for z,n in parsed if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id==orig for v in n.targets)]
            module=p.rsplit('.',1)[0].replace('.','/')
            rebindings=[(z,v) for z,v in rebindings if module in z.context.replace('\\','/')]
            if from_imp and rebindings and any(code_of(z).startswith('global '+orig) and module in z.context.replace('\\','/') for z in source):
                out.append(f"Packet line {r.number} uses `from {p.rsplit('.',1)[0]} import {orig} as {name}`."+
                           " The defining module shows "+"; ".join(f"`{orig} = {v}` (line {z.number})" for z,v in rebindings[:4])+
                           f". Reassigning its global name does not update the importing module's already-bound `{name}`.")
    for f,l in marked:
        tr=tree(l.code.strip())
        if not tr:continue
        for n in ast.walk(tr):
            if not isinstance(n,ast.Call):continue
            fn=previous.leaf(n.func)
            if fn=='where' and len(n.args)==3 and isinstance(n.args[0],ast.Compare) and all(isinstance(a,ast.Constant) for a in n.args[1:]):
                out.append(f"At packet line {l.physical}, the condition supplied to `{expr(n.func)}` is the Boolean RESULT `{expr(n.args[0])}`. The two other arguments are literals {expr(n.args[1])} and {expr(n.args[2])}. The comparison inputs and the mask supplied to the call are different expressions.")
            if fn in ('multiply','where') and len(n.args)>1 and all(not isinstance(a,ast.Constant) for a in n.args) and not isinstance(n.args[0],ast.Compare):
                if f.caret and previous.selected(f,l) is not None and expr(previous.selected(f,l))!=expr(n):continue
                out.append(f"Argument alternatives at packet line {l.physical}: "+", ".join(f"slot {i}: `{expr(a)}`" for i,a in enumerate(n.args))+". A caret on the whole call or a same-line primitive site does not distinguish a single argument.")
    for r,n in parsed:
        for q in ast.walk(n):
            if not isinstance(q,ast.Call) or previous.leaf(q.func) not in ('grad','value_and_grad','jacfwd','jacrev') or not q.args:continue
            alias=expr(q.func).split('.')[0]
            if not (expr(q.func).startswith('jax.') or any(p.startswith('jax.') for _,p,_,_ in imports.get(alias,[]))):continue
            if not isinstance(q.args[0],ast.Name) or len(defs.get(q.args[0].id,[]))!=1:continue
            rd,fn=defs[q.args[0].id][0]
            if not any(f.name.split('.')[-1]==fn.name for f,l in marked):continue
            arg=next((k.value for k in q.keywords if k.arg=='argnums'),q.args[1] if len(q.args)>1 else ast.Constant(value=0))
            try:values=ast.literal_eval(arg)
            except (ValueError,SyntaxError):continue
            if type(values) is int:values=(values,)
            formals=fn.args.posonlyargs+fn.args.args
            if isinstance(values,(tuple,list)) and values and all(type(i) is int and 0<=i<len(formals) for i in values):
                out.append(f"Source lines {rd.number} and {r.number}: `{expr(q)}` differentiates "+", ".join(f"`{formals[i].arg}` (zero-based parameter {i})" for i in values)+".")
    return list(dict.fromkeys(out))[:5]

def direct_bindings(source,t):
    sig=defaultdict(set)
    for r in source:
        s=code_of(r)
        if not s.startswith('def ') or not s.endswith(':'):continue
        tr=tree(s+'\n    pass')
        if tr is None:continue
        d=tr.body[0]
        if isinstance(d,ast.FunctionDef):sig[d.name].add((r.context,tuple(a.arg for a in d.args.posonlyargs+d.args.args)))
    links=[]
    for i in range(1,len(t.frames)):
        caller,callee=t.frames[i-1:i+1];fn=callee.name.split('.')[-1]
        if not fn.isidentifier() or re.search(r'/(jax|jaxlib)/_src/',previous.normpath(callee.header.code)):continue
        header=previous.normpath(callee.header.code)
        names={args for context,args in sig[fn] if context.endswith(".py") and previous.normpath(context) in header}
        m=previous.IPY.fullmatch(callee.header.code.strip())
        if m and m[3]:
            tokens=tuple(n.strip() for n in m[3].split(','))
            if tokens and all(n.isidentifier() for n in tokens):names={tokens}
        if len(names)!=1:continue
        formals=next(iter(names))
        for l in caller.lines:
            if not l.arrow:continue
            cs=[q for q in previous.calls(l.code) if previous.leaf(q.func)==fn]
            if len(cs)!=1:continue
            q=cs[0]
            if formals and formals[0] in ('self','cls'):
                if not isinstance(q.func,ast.Attribute) or (isinstance(q.func.value,ast.Name) and q.func.value.id[:1].isupper()):continue
                formals=formals[1:]
            positional=q.args;expanded=''
            if len(positional)==1 and isinstance(positional[0],ast.Starred) and isinstance(positional[0].value,ast.Name) and l.number:
                prev=[v for v in caller.lines if v.number==l.number-1]
                pt=tree(prev[0].code.strip()) if len(prev)==1 else None
                a=pt.body[0] if pt and len(pt.body)==1 else None
                if isinstance(a,ast.Assign) and len(a.targets)==1 and isinstance(a.targets[0],ast.Name) and a.targets[0].id==positional[0].value.id and isinstance(a.value,(ast.Tuple,ast.List)):
                    positional=a.value.elts;expanded=f" Expanded the immediately preceding tuple/list assignment at packet line {prev[0].physical}."
            if len(positional)>len(formals) or any(isinstance(a,ast.Starred) for a in positional) or any(k.arg is None for k in q.keywords):continue
            args=list(zip(formals,positional))+[(k.arg,k.value) for k in q.keywords if k.arg in formals]
            if not args:continue
            binds=', '.join(f"`{n}` ← `{expr(a)}`" for n,a in args[:7])
            if len(binds)<290:links.append(f"Caller line {l.physical} → next frame `{callee.name}`. Matching call and displayed signature: {binds}."+expanded)
    return links[-4:]

def producer_paths(rows,roots):
    assigned_rows=defaultdict(list);returns=defaultdict(list)
    active=None
    for r in rows:
        code=code_of(r);tr=tree(code)
        numbered=re.match(r'^\d+(?:: ?|  )(.*)$',r.code)
        indent=len(numbered[1])-len(numbered[1].lstrip()) if numbered else None
        if active and (active[0]!=r.context or indent is None or indent<=active[1]):active=None
        definition=re.fullmatch(r'def (\w+)\([^\n]*\)(?:\s*->[^:]*)?:',code)
        if definition and indent is not None:active=(r.context,indent,definition[1])
        if not tr or len(tr.body)!=1:continue
        n=tr.body[0]
        if isinstance(n,ast.Return) and active and n.value is not None:returns[(r.context,active[2])].append((r,expr(n.value)))
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            assigned_rows[n.targets[0].id].append((r,n.value))
    result=[]
    for name in sorted(roots,key=lambda s:(-len(s),s)):
        if len(assigned_rows[name])!=1:continue
        r,n=assigned_rows[name][0]
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name):
            matches=returns[(r.context,n.func.id)]
            if len(matches)==1:
                z,value=matches[0]
                result.append(f"Source lines {r.number} and {z.number}: `{name} = {expr(n)}`; the displayed same-file definition of `{n.func.id}` returns `{value}`.")
        for dep in sorted({a.id for a in ast.walk(n) if isinstance(a,ast.Name)}):
            if dep==name or len(assigned_rows[dep])!=1:continue
            prev,rhs=assigned_rows[dep][0]
            if prev.number>=r.number or not isinstance(rhs,(ast.Call,ast.Subscript,ast.Attribute,ast.BinOp)):continue
            term=expr(n);origin=expr(rhs)
            if len(term)>160 or len(origin)>160:continue
            result.append(f"Source lines {prev.number} → {r.number}: `{dep} = {origin}`; the expression in `{name} = {term}` uses the name `{dep}`.")
    return list(dict.fromkeys(result))[:4]

def eager_order(t):
    frames=list(previous.focal(t));facts=[]
    if not frames:return facts
    f,l=frames[0];s=l.code.strip();tr=tree(s)
    if tr and len(tr.body)==1:
        body=tr.body[0]
        value=getattr(body,'value',None)
        if isinstance(value,ast.Tuple) and len(value.elts)==1:value=value.elts[0]
        if isinstance(value,(ast.List,ast.Tuple)) and len(value.elts)>1 and isinstance(value.elts[0],ast.Call):
            q=value.elts[0]
            facts.append(f"List/tuple evaluation at packet line {l.physical} starts with `{expr(q)}` before the later item `{expr(value.elts[1])}`. This orders the candidates; it does not by itself prove which one raised.")
    for f,l in frames:
        for n in previous.calls(l.code):
            if isinstance(n.func,ast.Name) and n.func.id=='print' and len(n.args)==1 and isinstance(n.args[0],ast.Call):
                if frames[0][1].physical>l.physical:
                    facts.append(f"At packet line {l.physical}, `print` is waiting for `{expr(n.args[0])}` to return. The traceback is still inside that evaluation.")
    return facts

def index_roles(t):
    detail=" ".join(z.code.strip() for z in t.detail)
    m=re.search(r'depends on the values? of the arguments? (.+?)\.(?:\s|$)',detail)
    if not m:return []
    fields={v.strip(" '\"`") for v in re.split(r', | and ',m[1])}
    for f,l in previous.focal(t):
        tr=tree(l.code.strip())
        if not tr:continue
        lookups=[]
        for n in ast.walk(tr):
            if isinstance(n,ast.Subscript) and any(expr(q) in fields for q in ast.walk(n.slice) if isinstance(q,(ast.Name,ast.Attribute))):
                lookups.append(f"receiver `{expr(n.value)}`, index `{expr(n.slice)}`")
        if lookups:return [f"Operand roles at packet line {l.physical}: "+'; '.join(dict.fromkeys(lookups))+". The reported traced dependency occurs IN the index expression. These index expressions precede any use of the resulting entries."]
    return []

def composition_facts(source,t):
    result=[];statements=[]
    for r in source:
        tr=tree(code_of(r))
        if not tr or len(tr.body)!=1:continue
        n=tr.body[0];statements.append((r,n))
        for q in ast.walk(n):
            if not isinstance(q,ast.Call) or previous.leaf(q.func)!='Serial':continue
            for before,stage,after in zip(q.args,q.args[1:],q.args[2:]):
                if not isinstance(stage,ast.Call) or not isinstance(after,ast.Call) or previous.leaf(stage.func)!='Fn':continue
                lambdas=[a for a in stage.args if isinstance(a,ast.Lambda)]
                if len(lambdas)!=1 or not isinstance(lambdas[0].body,ast.Subscript):continue
                v=lambdas[0].body
                if not isinstance(v.slice,ast.Tuple) or any(not isinstance(a,ast.Slice) and not (isinstance(a,ast.Constant) and type(a.value) is int) for a in v.slice.elts):continue
                fixed=[i for i,a in enumerate(v.slice.elts) if isinstance(a,ast.Constant)]
                if len(fixed)!=1:continue
                result.append(f"Sequential stages at source line {r.number}: `{expr(before)}` → `{expr(v)}` → `{expr(after)}`. The integer index selects and removes axis {fixed[0]} while the slices retain the other axes. This gives the source of the following layer's input.")
    frames=list(t.frames_before_error());functions={f.name.split('.')[-1] for f,l in frames}
    if t.typ in ('TracerArrayConversionError','ConcretizationTypeError'):
        for i,(r,n) in enumerate(statements):
            if not isinstance(n,ast.Assign) or len(n.targets)!=1 or not isinstance(n.targets[0],ast.Name) or not isinstance(n.value,ast.Call):continue
            q=n.value
            if previous.leaf(q.func) not in ('NUTS','HMC') or not q.args or expr(q.args[0]) not in functions:continue
            for j,(s,m) in enumerate(statements[i+1:],i+1):
                if not isinstance(m,ast.Assign) or len(m.targets)!=1 or not isinstance(m.targets[0],ast.Name) or not isinstance(m.value,ast.Call) or previous.leaf(m.value.func)!='MCMC' or not m.value.args or expr(m.value.args[0])!=n.targets[0].id:continue
                for z,stmt in statements[j+1:]:
                    if any(isinstance(v,ast.Call) and expr(v.func)==m.targets[0].id+'.run' for v in ast.walk(stmt)):
                        result.append(f"Source call chain at packet lines {r.number}, {s.number}, {z.number}: `{expr(n)}` → `{expr(m)}` → `{expr(stmt)}`. The supplied model `{expr(q.args[0])}` also appears as a recorded frame.")
                        break
        focal=list(previous.focal(t))
        if focal:
            _,inner=focal[0]
            inputs=set()
            for q in previous.calls(inner.code):
                if previous.leaf(q.func) in previous.CONVERT:
                    inputs.update(a.id for a in q.args if isinstance(a,ast.Name))
            for _,l in focal[1:]:
                tr=tree(l.code.strip())
                if not tr or len(tr.body)!=1:continue
                n=tr.body[0]
                if not isinstance(n,ast.Assign) or not isinstance(n.value,ast.Call):continue
                overlap=inputs&assigned(l.code.strip())
                if overlap:result.append(f"At outer packet line {l.physical}, `{expr(n.value.func)}(...)` has not returned in this traceback. Its pending assignment to "+", ".join(f"`{x}`" for x in sorted(overlap))+f" cannot have supplied the inner operand at line {inner.physical} in this invocation.")
    return list(dict.fromkeys(result))[:3]

def card(flat,reviewed):
    t=Trace(flat)
    if t.error is None or t.typ not in ('TracerArrayConversionError','ConcretizationTypeError','TypeError'):return None
    if t.typ=='TypeError' and not (previous.SHAPE.fullmatch(t.message) or t.message.startswith('cannot reshape array of shape')):return None
    source=source_rows(flat);marked=list(reversed(list(previous.focal(t))))
    # Separate outward-to-inward stack order from a chain of value assignments.
    focus=marked if len(marked)<=7 else marked[:2]+marked[-5:]
    shapes,witness=source_shapes(source)
    details=shape_evidence(t,reviewed,source,shapes) if t.typ=='TypeError' else position_facts(t)+binding_facts(source,focus)+eager_order(t)+index_roles(t)
    details+=composition_facts(source,t)
    links=direct_bindings(source,t)
    seeds=set()
    for snippet in re.findall(r'`([^`\n]+)`',reviewed.splitlines()[0]):seeds|=ids(snippet)
    sink_seeds=set(seeds)
    if focus:sink_seeds.update(ids(focus[-1][1].code))
    for f,l in focus:
        seeds.update(ids(l.code.strip()))
    src=provenance(source,seeds,[f for f,_ in focus])
    producers=producer_paths(source,sink_seeds)
    for name,recs in witness.items():
        if name in seeds and name in shapes and len(recs)==1 and recs[0] not in src:src.append(recs[0])
    trace=[f"packet line {l.physical}, {f.name}: {l.code.strip()}" for f,l in focus if len(l.code.strip())<=380]
    facts=list(dict.fromkeys(details))
    parts=[]
    if trace:parts.append("Executed traceback sites, outermost to innermost (filtered frames can intervene)\n"+"\n".join(trace))
    if src:parts.append("Upstream source records selected by operand names, calls and definitions (source context, not proof of runtime identity)\n"+"\n".join(f"packet line {r.number}, {r.context}: {r.code}" for r in sorted(src,key=lambda z:z.number)))
    if links:parts.append("Caller-to-parameter matches (syntax checked against adjacent recorded frames)\n"+"\n".join(links))
    if producers:parts.append("Source producer expressions (displayed definitions; execution order is limited to what the traceback records)\n"+"\n".join(producers))
    parts.append("Reviewed trace relationship and its scope\n"+reviewed)
    if facts:parts.append("Computed operand and evaluation facts\n"+"\n".join("- "+d for d in facts))
    return dict(text="\n\n".join(parts),facts=facts,bindings=links,producers=producers,source_lines=[r.number for r in src],trace_lines=[l.physical for f,l in focus])
