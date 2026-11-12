"""Serve source evidence only when the marked sites have distinct frame headers."""
from .evidence import Trace, card, previous

def emit(flat, reviewed):
    trace=Trace(flat)
    if any(sum(bool(line.arrow) for line in frame.lines)>1 for frame in trace.frames):
        return "parent",reviewed
    result=card(flat,reviewed)
    return ("expanded",result["text"]) if result else ("parent",reviewed)

def analyze_packet(flat, reviewed_hint=None):
    """Analyze a new source packet, or extend a caller-supplied reviewed hint."""
    if reviewed_hint is None:
        raw=previous.v1.base.analyze(flat)
        if raw["relationship"]:
            prior=previous.v1.render_previous(flat,raw["relationship"],raw)
        else:
            prior=previous.propose(flat) or previous.v1.expansion(flat)
        if not prior:return None
        reviewed_hint=prior["hint"]
    method,text=emit(flat,reviewed_hint)
    return dict(method=method,evidence=text)
