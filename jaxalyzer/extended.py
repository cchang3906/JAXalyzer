"""Additional source/trace joins for the previously uncovered part of the cohort."""
import ast
from dataclasses import dataclass,field
import io
import math
import re
import tokenize

from . import hints as v1

base=v1.base
Line=base.Line
ANSI=re.compile(r"\x1b\[[\d;]*m")
NATIVE=re.compile(r'^File\s+["“]?(.*?)["”]?,\s+line\s+(\d+),\s+in\s+(.+?)\s*$')
IPY=re.compile(r'^(?:File\s+)?(.+?)\s+in\s+([\w.<>]+)(?:\((.*)\))?\s*$')
HEADER=re.compile(r'^(?:[\w.]+\.)?(TracerArrayConversionError|ConcretizationTypeError|TypeError):\s*(.*)$')
SHAPE=re.compile(r'dot_general requires contracting dimensions to have the same shape, got\s*[\[(](\d+),?[\])] and [\[(](\d+),?[\])][.]?$')
PRIMITIVE=re.compile(r'^operation\s+\w+\s*:\s*[\w]+\[[^\]]*\]\s*=\s*(\w+)')
ORIGIN=re.compile(r'^(?:This\s+[\w.]*Tracer(?: with object id [\w\[\]-]+)? was created on line[: ]*|from line )(.+?)\s*\(([\w.<>]+)\)\s*$')
TRANSFORM=re.compile(r"^(?:The error occurred while tracing|While tracing) the function ([\w.<>]+) at (.+?)(?: for (\w+))?[.,] (?:[Tt]his (?:concrete )?value|$)")
TRANSFORMS={"grad","jit","vmap","pmap","jacfwd","jacrev","value_and_grad","scan","map"}
CONVERT={"array","asarray","asanyarray","float","int","sqrt","cos","sin","log","exp","where","tril","multiply","equal","item"}
SHAPE_OPS={"dot","matmul","tensordot","dot_general","einsum"}

@dataclass
class Frame:
    name:str
    header:Line
    lines:list=field(default_factory=list)
    native:bool=False
    caret:tuple|None=None

def clean_code(s):
    return re.sub(r'^L\d+: ?', '',ANSI.sub('',s)).replace('“','"').replace('”','"')

def normpath(s):
    return s.replace("\\","/").lower()

def parse_tree(s):
    try:return ast.parse(s.strip())
    except (SyntaxError,ValueError):return None

def calls(s):
    node=parse_tree(s)
    return [n for n in ast.walk(node) if isinstance(n,ast.Call)] if node else []

def leaf(n):
    return base.callable_name(n)

def pos(n):return (n.col_offset,n.end_col_offset)

def call_fragments(s,allowed):
    """Extract complete calls from truncated context; tokenize excludes strings/comments."""
    tokens=[]
    try:
        for t in tokenize.generate_tokens(io.StringIO(s.strip()).readline):
            if t.type in (tokenize.NAME,tokenize.OP,tokenize.NUMBER,tokenize.STRING):
                tokens.append(t)
    except (tokenize.TokenError,IndentationError,SyntaxError):pass
    out=[]
    for i,t in enumerate(tokens[:-1]):
        if t.type!=tokenize.NAME or t.string not in allowed or tokens[i+1].string!="(":continue
        start=i
        while start>=2 and tokens[start-1].string=="." and tokens[start-2].type==tokenize.NAME:start-=2
        level=0
        for u in tokens[i+1:]:
            if u.string in ('(','[','{'):level+=1
            elif u.string in (')',']','}'):level-=1
            if level==0:
                raw=s.strip()[tokens[start].start[1]:u.end[1]]
                try:node=ast.parse(raw,mode="eval").body
                except (ValueError,SyntaxError):break
                if isinstance(node,ast.Call):out.append(node)
                break
    return out

class Trace:
    def __init__(self,text):
        lines=[Line(clean_code(s),i,"recorded traceback") for i,s in enumerate(text.splitlines(),1)]
        self.records={l.physical:l for l in lines}
        starts=[i for i,l in enumerate(lines) if re.fullmatch(r'RECORDED TRACEBACK(?: \([^()]+\))?',l.code.strip())]
        self.frames=[];self.sources=[];self.detail=[];self.error=None;self.typ=None;self.message=""
        if len(starts)!=1:return
        start=starts[0]+1
        stop=next((i for i in range(start,len(lines)) if re.match(r"^(?:SOURCE CONTEXT|SOURCE EXCERPTS|EVIDENCE LIMITS|SOURCE ATTRIBUTION|DERIVED RELATIONSHIPS)",lines[i].code)),len(lines))
        trace=lines[start:stop]
        chain=[i for i,l in enumerate(trace) if l.code.strip() in ("During handling of the above exception, another exception occurred:","The above exception was the direct cause of the following exception:")]
        if chain:
            cut=chain[-1]+1
            while cut<len(trace) and not trace[cut].code.strip():cut+=1
            if cut>=len(trace) or trace[cut].code.strip()!="Traceback (most recent call last):":return
            trace=trace[cut:]
        matches=[(i,HEADER.fullmatch(l.code.strip())) for i,l in enumerate(trace) if HEADER.fullmatch(l.code.strip())]
        if len(matches)!=1:return
        ei,em=matches[0];self.error=trace[ei];self.typ,self.message=em.groups()
        if any(re.match(r'^(?:[\w.]+\.)?\w*(?:Error|Exception)(?::|$)',l.code.strip()) and not l.code.strip().startswith('LayerError: Exception passing through layer') for l in trace[:ei]):self.error=None;return
        if any(l.code.strip().startswith("Traceback (most recent call last)") or re.match(r'^(?:[\w.]+\.)?\w*(?:Error|Exception)(?::|$)',l.code.strip()) for l in trace[ei+1:]):self.error=None;return
        for l in trace[ei+1:]:
            ss=l.code.strip()
            if re.match(r'^(?:[\w.]+\.)?\w*(?:Error|Exception)(?::|$)',ss) or ss.startswith('Traceback ('):
                self.error=None;return
            if ss.startswith(('See https://','(https://','--------------------')):break
            if ss:self.detail.append(l)
        last=None
        for line in trace[:ei]:
            s=line.code; stripped=s.strip()
            n=NATIVE.fullmatch(stripped)
            archived=IPY.fullmatch(stripped)
            if n or (archived and (re.search(r'\.py[: ]|<ipython|Cell In|Input In',stripped) or stripped.startswith("File "))) or re.match(r'^Cell In\s*\[\d+\], line \d+',stripped):
                name=n[3] if n else archived[2] if archived else "<module>"
                last=Frame(name,line,native=bool(n));self.frames.append(last);continue
            if stripped.startswith("│") and re.search(r'\bin\s+\S+',stripped) and '❱' not in stripped:
                last=Frame(stripped,line);self.frames.append(last);continue
            if not last or not stripped:continue
            tag=re.match(r'^\s*(?:(-+>)\s*)?(\d+)\s+(.*)$',s)
            rich=re.match(r'^\s*│\s*❱\s*(\d+)\s*(.*?)\s*│\s*$',s)
            if tag or rich:
                num=int(tag[2] if tag else rich[1]);code=tag[3] if tag else rich[2].lstrip(' │')
                last.lines.append(Line(code,line.physical,last.name,num,bool(rich or tag[1])))
            elif last.native and not last.lines and not stripped.startswith(('...', '│','[...')):
                last.lines.append(Line(s.lstrip(),line.physical,last.name,int(NATIVE.fullmatch(last.header.code.strip())[2]),True))
            elif last.native and last.lines and re.fullmatch(r'[~^]+',stripped):
                prev=last.lines[-1]
                if line.physical==prev.physical+1 and self.records[prev.physical].code.isascii() and '\t' not in self.records[prev.physical].code:
                    raw=self.records[prev.physical].code
                    col=len(s)-len(s.lstrip())-(len(raw)-len(raw.lstrip()))
                    if col>=0:last.caret=(col,col+len(stripped))
        source=False
        for l in lines[stop:]:
            if l.code.startswith(("SOURCE CONTEXT","SOURCE EXCERPTS","SUPPLEMENTAL RECORDED TRACEBACK")):source=True;continue
            if l.code.startswith(("SOURCE ATTRIBUTION","EVIDENCE LIMITS","DERIVED RELATIONSHIPS")):source=False
            if l.code.startswith(("Attribution:","Source:","License:")):continue
            if not source:continue
            m=re.match(r'^\s*(?:L?(\d+):\s*|(?:-+>\s*)?(\d+)\s{1,})(.*)$',l.code)
            self.sources.append(Line((m[3] if m else l.code).strip(),l.physical,"displayed source",int(m[1] or m[2]) if m else None))
    def frames_before_error(self):
        return [(f,l) for f in self.frames for l in f.lines if l.arrow and l.code.strip() and not l.code.strip().startswith(('raise ','[...]','...'))]

def pack(kind,claim,limit,refs):
    return v1.make(kind,claim,limit,refs)

def witness_text(t):
    details=t.detail
    for i,l in enumerate(details):
        if 'was created on line' in l.code:
            raw=l.code.strip()
            used=[l]
            if raw.endswith(':') and i+1<len(details):
                raw+=' '+details[i+1].code.strip();used.append(details[i+1])
            m=ORIGIN.fullmatch(raw)
            if m and re.search(r':\d+(?::\d+)?$',m[1]):
                return f"The error report places creation of the tracer at `{m[1].strip()}` in `{m[2]}`.",used
    return '',[]

def provenance(t):
    witnesses=[]
    for i,l in enumerate(t.detail[:-1]):
        op=PRIMITIVE.match(l.code.strip())
        m=ORIGIN.fullmatch(t.detail[i+1].code.strip())
        if not op or not m:continue
        loc=re.search(r':(\d+)(?::(\d+))?$',m[1])
        if not loc:continue
        no=int(loc[1]);fn=m[2]
        location=normpath(m[1][:loc.start()]).rsplit('/',1)[-1]
        matches=[(f,l2) for f,l2 in t.frames_before_error() if f.name.split('.')[-1]==fn and l2.number==no and (location in normpath(f.header.code) or 'cell in[' in normpath(f.header.code) and '/ipykernel_' in normpath(m[1]))]
        if not matches:
            matches=[(None,l2) for l2 in t.sources if l2.number==no and parse_tree(l2.code) and not l2.code.startswith(("def ","#"))]
            # Supplemental source needs both the matching function header and matching line number.
            matches=[(f,l2) for f,l2 in matches if any(re.match(r'def '+re.escape(fn)+r'\(',z.code) and z.physical<l2.physical and z.number is not None and 0<l2.number-z.number<=l2.physical-z.physical for z in t.sources) and any(location in normpath(z.code) and (' in '+fn+'(') in z.code and 0<l2.physical-z.physical<10 for z in t.sources)]
        if len(matches)==1:
            ln=matches[0][1]
            witnesses.append((op[1],fn,no,ln,l,t.detail[i+1]))
    if not witnesses:return None
    op,fn,no,ln,rep,site=witnesses[0]
    what=ln.code.strip()
    if len(what)>320:return None
    downstream=next(((f,l) for f,l in focal(t) if l.physical>ln.physical and top_site(l)),None)
    context=f" The later marked frame in `{downstream[0].name}` shows `{top_site(downstream[1])}`." if downstream else ""
    nested=[n for n in calls(what) if leaf(n.func) in CONVERT and len(n.args) and isinstance(n.args[0],ast.Call)]
    if len(nested)==1:
        n=nested[0]
        context+=f" In that source line the result of `{base.expr(n.args[0])}` is supplied directly to `{base.expr(n.func)}`."
    return pack("reported_primitive_source",f"The error report attributes a `{op}` operation to line {no} of `{fn}`. The displayed line with that function name and line number is `{what}`."+context,
                "A reported creation site can differ from the later operation requesting conversion. A whole-line match does not establish which subexpression produced the tracer.",[t.error,ln,rep,site]+([downstream[1]] if downstream else []))

def focal(t):
    for f,l in reversed(t.frames_before_error()):
        path=normpath(f.header.code)
        if re.search(r'/(?:site|dist)-packages/(?:jax|jaxlib)/|/jax/_src/',path):continue
        yield f,l

def selected(f,l):
    if not f.caret:return None
    s=l.code.strip()
    for p,q in [('',''),('{','}'),('(',')')]:
        tree=parse_tree(p+s+q)
        if not tree:continue
        found=[n for n in ast.walk(tree) if isinstance(n,(ast.Call,ast.BinOp)) and (n.col_offset-len(p),n.end_col_offset-len(p))==f.caret]
        if len(found)==1:return found[0]
    return None

def aliases(t):
    for l in t.sources:
        tree=parse_tree(l.code)
        if not tree or len(tree.body)!=1 or not isinstance(tree.body[0],ast.Assign):continue
        a=tree.body[0];q=a.value
        if len(a.targets)==1 and isinstance(q,ast.Call) and leaf(q.func) in TRANSFORMS and q.args and isinstance(q.args[0],(ast.Attribute,ast.Name)):
            yield l,base.expr(a.targets[0]),q

def mapping(t):
    for f,line in focal(t):
        m=IPY.match(f.header.code.strip())
        if not m or not m[3]:continue
        names=[s.strip() for s in m[3].split(',')]
        if names and names[0] in ('self','cls'):names=names[1:]
        possible=[(z,n,q) for z,n,q in aliases(t) if any(base.expr(c.func)==n for _,u in t.frames_before_error() if u.physical<line.physical for c in calls(u.code))]
        for _,u in t.frames_before_error():
            if u.physical<line.physical:possible.extend((u,None,q) for q in call_fragments(u.code,{'vmap'}))
        for z,name,q in possible:
            if leaf(q.func)!='vmap' or not q.args or base.expr(q.args[0]).split('.')[-1]!=f.name.split('.')[-1]:continue
            axes=next((k.value for k in q.keywords if k.arg=='in_axes'),None)
            if not isinstance(axes,(ast.List,ast.Tuple)) or len(axes.elts)!=len(names):continue
            for op in calls(line.code):
                if leaf(op.func) not in CONVERT or len(op.args)!=1 or not isinstance(op.args[0],ast.Name) or op.args[0].id not in names:continue
                n=op.args[0].id;i=names.index(n);v=axes.elts[i]
                if not isinstance(v,ast.Constant) or type(v.value) is not int:continue
                return pack("mapped_operand",f"The displayed `vmap` of `{base.expr(q.args[0])}` gives axis {v.value} to `{n}` when its positional axes are aligned with the displayed signature. The marked frame passes that parameter to `{base.expr(op)}`.",
                    "This matches the transform specification to the source parameter. It does not recover a concrete value or the full runtime shape.",[z,f.header,line,t.error])
    return None

def caret_link(t):
    frames=list(focal(t))
    for f,l in frames[:1]:
        if not f.caret:continue
        text=l.code.strip(); prefix=''
        tree=parse_tree(text)
        if not tree:
            for start,end in [('{' ,'}'),('(',')')]:
                tree=parse_tree(start+text+end)
                if tree:prefix=start;break
        if not tree:continue
        matched=[n for n in ast.walk(tree) if isinstance(n,(ast.Call,ast.BinOp)) and (n.col_offset-len(prefix),n.end_col_offset-len(prefix))==f.caret]
        if len(matched)!=1:continue
        n=matched[0];claimed='';extra_refs=[]
        if len(frames)>1 and isinstance(n,ast.Call) and leaf(n.func) in CONVERT and len(n.args)==1:
            parent,up=frames[1];delegating=selected(parent,up)
            if isinstance(delegating,ast.Call) and leaf(delegating.func)==f.name and len(delegating.args)==1 and t.frames.index(f)==t.frames.index(parent)+1:
                claimed=f"The caller caret selects `{base.expr(delegating)}`. The next frame, in `{f.name}`, marks `{base.expr(n)}`. The two positions connect that caller argument to the conversion path."
                extra_refs=[up,t.records[up.physical+1]]
        if not claimed and isinstance(n,ast.Call) and leaf(n.func) in CONVERT and n.args and isinstance(n.args[0],(ast.BinOp,ast.Call,ast.Subscript)):
            claimed=f"The traceback caret selects `{base.expr(n)}`. Its first argument is the result of `{base.expr(n.args[0])}`."
            if len(n.args)==1:claimed+=" This distinguishes the operand from the individual names inside the argument."
        elif not claimed and isinstance(n,ast.Call) and leaf(n.func) in CONVERT and len(n.args)>1:
            claimed=f"The traceback caret selects the call `{base.expr(n)}`; its displayed positional inputs are "+", ".join(f"`{base.expr(a)}`" for a in n.args)+"."
        elif isinstance(n,ast.BinOp) and isinstance(n.op,ast.Mod) and isinstance(n.left,ast.Constant) and isinstance(n.left.value,str) and isinstance(n.right,ast.Tuple):
            formats=re.findall(r'%(?:[-+ 0#]*\d*(?:\.\d+)?)?([sdfeg])',n.left.value,re.I)
            if len(formats)==len(n.right.elts) and sum(x.lower() in "feg" for x in formats)==1 and '%(' not in n.left.value and '%%' not in n.left.value:
                ix=next(i for i,x in enumerate(formats) if x.lower() in "feg")
                claimed=f"The caret selects the percent-formatting expression inside the displayed call. Its floating-point format field receives `{base.expr(n.right.elts[ix])}`, before the enclosing function can receive the formatted string."
        if not claimed:continue
        detail=' '.join(z.code.strip() for z in t.detail)
        transform=TRANSFORM.search(detail)
        extra=''
        if transform: extra=f" The diagnostic names `{transform[1]}` as the function being traced."
        elif t.typ=="ConcretizationTypeError" and 'problem arose with the `float`' in detail:extra=" JAX attributes the concretization to a Python float conversion."
        else:
            upstream=[(u,v) for u,v in t.frames_before_error() if v.physical<l.physical and re.search(r'\b(?:jax\.)?(grad|jit|vmap|jacfwd)\(',v.code) and not v.code.lstrip().startswith(('#','"'))]
            if upstream:
                code=upstream[-1][1]
                if len(code.code.strip())<=220:extra=f" An earlier recorded frame shows `{code.code.strip()}`."
        support=t.detail[:2] if transform or 'float conversion' in extra else [code] if extra else []
        links=[(z,name,q) for z,name,q in aliases(t) if base.expr(q.args[0]).split('.')[-1]==f.name.split('.')[-1]]
        if len(links)==1:
            z,name,q=links[0]
            extra+=f" The source binds `{name}` to `{base.expr(q)}`, targeting this function.";support.append(z)
        return pack("positioned_operand",claimed+extra,"The caret localizes the displayed operation; it does not recover earlier runtime values or distinguish among several converted arguments of a single call.",extra_refs+[l,t.records[l.physical+1],t.error]+support)
    return None


def published_transform(t):
    for f,line in reversed(t.frames_before_error()):
        if '/jax/' in normpath(f.header.code):continue
        text=line.code.strip();tree=parse_tree(text)
        if not tree:continue
        candidates=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and (leaf(n.func) in CONVERT or isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Call))]
        candidates=[n for n in candidates if n.args or leaf(n.func)=="item"]
        lambda_call=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and any(isinstance(x,ast.Lambda) for x in n.args)]
        if not candidates and not lambda_call:continue
        site=base.expr(candidates[0]) if len(candidates)==1 else line.code.strip() if candidates else base.expr(lambda_call[0])
        bindings=[]
        for l in t.sources:
            tr=parse_tree(l.code)
            if not tr or len(tr.body)!=1 or not isinstance(tr.body[0],ast.Assign):continue
            a=tr.body[0]
            if not isinstance(a.value,ast.Call) or leaf(a.value.func) not in TRANSFORMS or not a.value.args or len(a.targets)!=1:continue
            name=base.expr(a.targets[0]);fun=base.expr(a.value.args[0])
            for fr,ln in t.frames_before_error():
                if any(base.expr(n.func)==name for n in calls(ln.code)) and ln.physical<line.physical and any(g.name.split('.')[-1]==fun.split('.')[-1] for g in t.frames):
                    bindings.append((l,ln,base.expr(a.value),name))
        if len({(x[2],x[3]) for x in bindings})==1:
            l,up,transform,name=bindings[0]
            return pack("transformed_call_path",f"The displayed source binds `{name}` to `{transform}`. The recorded stack calls `{name}` and then shows `{site}` in `{f.name}` on the conversion path.",
                        "The saved binding and the traceback connect the transformation to the failing path. They do not prove a particular scalar value or an unshown library implementation.",[l,up,line,t.error])
        upstream=[]
        for fr,ln in t.frames_before_error():
            if ln.physical>=line.physical or '/jax/' in normpath(fr.header.code):continue
            cs=call_fragments(ln.code,TRANSFORMS)
            for call in cs:
                if not call.args:continue
                arg=call.args[0]
                if not (isinstance(arg,(ast.Name,ast.Attribute,ast.Lambda)) and (isinstance(arg,ast.Lambda) or any(ff.name.split('.')[-1]==base.expr(arg).split('.')[-1] for ff in t.frames))):continue
                if not (base.expr(call.func).startswith(('jax.','lax.')) or any(re.search(r'\bfrom jax import\b.*\b'+re.escape(leaf(call.func))+r'\b',x.code) for x in t.sources)):continue
                upstream.append((ln,call))
        if upstream:
            up,call=upstream[-1]
            return pack("transformed_call_path",f"The marked stack contains `{base.expr(call)}` and reaches `{site}` in `{f.name}` before the reported conversion error.",
                        "This is a connection along the recorded call path. The transform on that path does not by itself establish the exact traced argument or which nested operation first converted it.",[up,line,t.error])
    return None

def top_site(line):
    tree=parse_tree(line.code)
    found=[] if tree is None else [n for n in ast.walk(tree) if isinstance(n,ast.Call) and leaf(n.func) in CONVERT]
    if not found:found=call_fragments(line.code,CONVERT)
    if found:
        return base.expr(found[0]) if len(found)==1 else line.code.strip()
    if tree and any(isinstance(n,ast.Subscript) for n in ast.walk(tree)):
        return line.code.strip()
    return None

def report_path(t):
    summary=" ".join(l.code.strip() for l in t.detail)
    match=re.search(r'(?:While tracing|The error occurred while tracing) the function ([\w.<>]+) at .+? for ([\w]+)[.,]',summary)
    dep=re.search(r'it depends on the values? of the arguments? (.+?)\.(?:\s|$)',summary)
    provenance,refs=witness_text(t)
    if not match and not provenance:return None
    if match and not (dep or 'became a tracer due to JAX operations' in summary) and not provenance:return None
    for f,l in focal(t):
        site=top_site(l)
        if not site or len(site)>350:continue
        if match:
            evidence=f"The diagnostic identifies `{match[1]}` under `{match[2]}`"
            if dep:evidence+=f" and names {dep[1]} as an input dependency"
            evidence+="."
        else:evidence=provenance
        if match and provenance:evidence+=" "+provenance
        if dep:
            fields={w.strip(" '\"") for w in re.split(r', | and ',dep[1])}
            tree=parse_tree(l.code)
            if tree:
                idx=sorted({base.expr(n.slice) for n in ast.walk(tree) if isinstance(n,ast.Subscript) and any(base.expr(w) in fields for w in ast.walk(n.slice) if isinstance(w,(ast.Attribute,ast.Name)))})
                if idx:evidence+=" On that source line "+", ".join(f"`{w}`" for w in idx)+" occurs in an index expression involving the named dependency."
        return pack("tracing_context_to_use",evidence+f" The displayed conversion path reaches `{site}` in `{f.name}`.",
                    "This relates the reported tracing or creation context to a use site. It does not prove that every operand is traced or that a creation site itself requested conversion.",[l,t.error]+refs+[z for z in t.detail if "tracing the function" in z.code or "depends on" in z.code])
    if match:
        # Some archived diagnostics retain the transform call and source but no consuming traceback frame.
        for l in t.sources:
            tr=parse_tree(l.code)
            if not tr or len(tr.body)!=1 or not isinstance(tr.body[0],ast.Assign):continue
            a=tr.body[0];q=a.value
            if not isinstance(q,ast.Call) or base.expr(q.func)!="jax."+match[2] or not q.args or base.expr(q.args[0])!=match[1]:continue
            claim=f"The report identifies `{match[1]}` as the function traced for `{match[2]}`. The displayed source constructs that wrapper with `{ast.unparse(a)}`."
            if dep:claim+=f" The diagnostic names {dep[1]} as a dependency of the abstract value."
            if any('problem arose with the `float`' in x.code for x in t.detail):
                items=[x for x in t.sources if any(leaf(n.func)=="item" for n in calls(x.code))]
                if len(items)==1:
                    claim+=f" The excerpt also shows scalar extraction in `{items[0].code}`; the error names a float conversion."
                    return pack("tracing_context_to_use",claim,"The excerpt identifies an extraction site without a frame selecting that line as the unique conversion.",[t.error,l,items[0]]+[x for x in t.detail if 'function' in x.code][:2])
            if dep:return pack("tracing_context_to_use",claim,"This matches the wrapper and reported dependency; the available source does not establish the exact converted subexpression.",[t.error,l]+[x for x in t.detail if 'function' in x.code or 'depends on' in x.code])
    return None

def source_definitions(t,names):
    bindings={}
    for l in t.sources:
        if not l.code or l.code.startswith(('#','def ','class ')):continue
        for name in names:
            pattern=r'(?<![\w.])'+re.escape(name)+r'\s*=\s*([A-Za-z_]\w*\[[A-Za-z_]\w*\])(?=[\s,;.)]|$)'
            for m in re.finditer(pattern,l.code):
                bindings.setdefault(name,[]).append((m[1],l))
            m=re.search(r'(?<![\w.])'+re.escape(name)+r' is (\[\[[\d ,\[\]-]+\]\])',l.code)
            if m:
                try:v=ast.literal_eval(m[1])
                except (SyntaxError,ValueError):continue
                if v and all(isinstance(r,list) and len(r)==len(v[0]) for r in v):
                    bindings.setdefault(name,[]).append((f"matrix with shape ({len(v)}, {len(v[0])})",l))
    return [(name,refs[0][0],refs[0][1]) for name,refs in sorted(bindings.items()) if len({v for v,_ in refs})==1]

def shape(t):
    dm=SHAPE.fullmatch(t.message)
    if dm:
        if dm[1]==dm[2]:return None
        # Prefer the last displayed application contraction before a matching library implementation.
        for f,l in focal(t):
            path=normpath(f.header.code)
            if '/_src/numpy/' in path or '/_src/lax/' in path:continue
            tree=parse_tree(l.code)
            ops=[] if tree is None else [n for n in ast.walk(tree) if isinstance(n,ast.BinOp) and isinstance(n.op,ast.MatMult) or isinstance(n,ast.Call) and leaf(n.func) in SHAPE_OPS]
            if tree and any(isinstance(n,(ast.Lambda,ast.IfExp,ast.BoolOp,ast.comprehension)) for n in ast.walk(tree)):continue
            if not ops:ops=call_fragments(l.code,SHAPE_OPS)
            if not ops:continue
            later=[ff for ff in t.frames if ff.header.physical>l.physical]
            if not any('/_src/lax/' in normpath(ff.header.code) and ff.name=='_dot_general_shape_rule' or '/_src/numpy/' in normpath(ff.header.code) and ff.name in ('dot','matmul','tensordot') for ff in later):return None
            names={n.id for op in ops for n in ast.walk(op) if isinstance(n,ast.Name)}
            binds=source_definitions(t,names)
            if len(ops)>1:
                if not binds or not all(isinstance(x,ast.BinOp) and isinstance(x.op,ast.MatMult) for x in ops):return None
                claim="The marked line includes "+' and '.join('`'+base.expr(x)+'`' for x in ops)+". The source reconstruction gives "+', '.join(f"`{n}` as `{v}`" for n,v,_ in binds)+f". The internal contraction reports extents {dm[1]} and {dm[2]}."
                return pack("contraction_source_bindings",claim,"These source bindings help distinguish the candidate products. The traceback does not mark an individual product or establish that all saved shapes were live.",[l,t.error]+[x for _,_,x in binds])
            op=ops[0]; method='matmul' if isinstance(op,ast.BinOp) else leaf(op.func)
            if isinstance(op,ast.Call):
                if method not in ('dot','matmul','tensordot') or len(op.args)<2 or any(isinstance(n,ast.Starred) for n in op.args):return None
                if base.expr(op.func).rsplit('.',1)[0] not in ('jnp','np','jax.numpy','numpy'):return None
                a,b=map(base.expr,op.args[:2])
            else:a,b=base.expr(op.left),base.expr(op.right)
            matched=[ff for ff in later if ff.name==method and '/_src/numpy/' in normpath(ff.header.code)]
            if not matched and not any(ff.name=='_dot_general_shape_rule' for ff in later):return None
            if any('/jax/' not in normpath(ff.header.code) and '/_src/' not in normpath(ff.header.code) and not ('deferring_binary_op' in ff.name) for ff in later):
                return None
            if method=='tensordot':
                if len(op.args)<3:return None
                try:axes=ast.literal_eval(op.args[2])
                except (ValueError,SyntaxError):return None
                if not isinstance(axes,(list,tuple)) or len(axes)!=2 or not all(type(i) is int for i in axes):return None
                axes_text=f"The supplied axes select axis {axes[0]} of `{a}` and axis {axes[1]} of `{b}`."
            else:axes_text=f"Under the array `{method}` contract, the final axis of `{a}` must match the penultimate axis of `{b}` (its only axis for a vector)."
            refs=[l,t.error]+[ff.header for ff in matched[:1]]
            return pack("contraction_source_bindings",f"The marked expression is `{base.expr(op)}`. {axes_text} The recorded lower-level contraction has incompatible extents {dm[1]} and {dm[2]}.",
                        "The reported pair describes contracting extents. It does not give both full shapes, the arrays' earlier history, or the runtime identity of a source alias.",refs)
        return None
    pat=r'cannot reshape array of shape ('+v1.DIM+r') \(size (\d+)\) into shape ('+v1.DIM+r') (?:\(size (\d+)\)|because the product of specified axis sizes \((\d+)\) does not evenly divide (\d+))[.]?'
    m=re.fullmatch(pat,t.message)
    if not m:return None
    try:src,dest=v1.dims(m[1]),v1.dims(m[3])
    except (SyntaxError,ValueError,TypeError):return None
    if not src or not dest or any(type(x) is not int or x<0 for x in src) or any(type(x) is not int or x< -1 for x in dest) or dest.count(-1)>1 or math.prod(src)!=int(m[2]):return None
    fixed=math.prod(x for x in dest if x!=-1)
    if m[4]:
        if -1 in dest or int(m[4])!=fixed or int(m[2])==fixed:return None
    elif -1 not in dest or int(m[5])!=fixed or fixed==0 or int(m[6])!=int(m[2]) or int(m[2])%fixed==0:return None
    for f,l in focal(t):
        if not any(ff.name=='_compute_newshape' and '/_src/numpy/' in normpath(ff.header.code) and ff.header.physical>l.physical for ff in t.frames):continue
        op=call_fragments(l.code,{'reshape'})
        if len(op)!=1 or not op[0].args or op[0].keywords:continue
        op=op[0]
        if len(op.args)>1 and len(op.args)==len(dest):
            if any(isinstance(a,ast.Constant) and a.value!=n for a,n in zip(op.args,dest)):continue
            slots=', '.join(f"`{base.expr(a)}` at target axis {i} is {n}" for i,(a,n) in enumerate(zip(op.args,dest)) if isinstance(a,ast.Name))
            if not slots:continue
        elif len(op.args)==1:
            target=op.args[0]
            if isinstance(target,ast.Constant) and (len(dest)!=1 or target.value!=dest[0]):continue
            if isinstance(target,ast.Attribute) and target.attr=='shape':
                slots=f"the shown target `{base.expr(target)}` corresponds to {dest} in the recorded reshape"
                complete=[n for n in calls(l.code) if isinstance(n.func,ast.Attribute) and n.func.attr=='reshape']
                if len(complete)==1:slots+=f" and its receiver is `{base.expr(complete[0].func.value)}`"
            elif isinstance(op.func,ast.Attribute) and isinstance(op.func.value,ast.Name):
                slots=f"the receiver `{base.expr(op.func.value)}` has recorded shape {src} at `{base.expr(op)}`"
            else:continue
        else:continue
        return pack("reshape_source_constraint",f"In the marked reshape, {slots}. The recorded operand contains {math.prod(src)} elements; "+(f"the target requires {fixed}." if m[4] else f"the fixed axes require a multiple of {fixed} before the −1 axis can be inferred."),
                    "These are use-site sizes. They do not establish the original caller's shape or which earlier assignment introduced the mismatch.",[l,t.error])
    return None

def propose(flat):
    t=Trace(flat)
    if not t.error:return None
    if t.typ=='TypeError':return shape(t)
    if t.typ not in ('ConcretizationTypeError','TracerArrayConversionError'):return None
    return caret_link(t) or provenance(t) or mapping(t) or report_path(t) or published_transform(t) or v1.expansion(flat)


