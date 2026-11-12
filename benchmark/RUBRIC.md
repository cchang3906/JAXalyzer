# Diagnostic scoring: proposed pre-inference rubric

Question for both arms: “Identify the exact operation that raises, the immediate operand or constraint responsible, and its source or origin. Explain why the execution fails. Cite a line or expression. If the evidence does not determine a detail, say so. Do not propose a patch.”

Before inference, freeze the key, supporting lines, acceptable equivalent answers and any details the packet cannot establish. Score the operation that actually raised, not a nearby operation or a suggested fix. Do not require an unstated axis, runtime value, call binding or reason. If the shown packet cannot support the required diagnosis, withhold it before launch and record why.

Grade each answer independently under an opaque answer ID. The grader sees the question, relevant source/trace and frozen key, never the model, arm, mate's answer, or dispatch order.

| Grade | Rule |
|---|---|
| Correct | Identifies the failing operation and correct operand or conflicting axes, explains the cause to the depth specified in the key, and cites matching source or traceback. No conflicting causal assertion. Equivalent wording or line numbering is fine. |
| Partial | Identifies a true part of the diagnosis, but misses a required element or fails to cite evidence. |
| Incorrect | Blames the wrong operation, operand or axis, contradicts a required fact, or provides only general advice. |
| Unavailable | No usable answer was received inside the sealed request limit. Preserve the reason and timing; no re-prompt. |

The primary paired measure is the change in **timely correct diagnosis**, derived minus flat, for each model separately. Publish derived-only and flat-only wins, both correct, neither correct, unavailable arms, and a sensitivity result restricted to pairs with both answers. Show task totals and group/family counts; report previously inspected examples and baseline-rule vs. newly covered cases separately. Do not pool models as new cases. Specificity controls and unkeyed cases do not enter the model accuracy denominator. The earlier PR study keeps its own frozen rubric and thresholds.
