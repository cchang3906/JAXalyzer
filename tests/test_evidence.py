import unittest
from jaxalyzer.evidence import shape_of, dimension_facts, position_facts, source_rows, provenance, card, Trace, direct_bindings, producer_paths, binding_facts

def packet(code, error, source="", caret=None):
    frame=f'  File "/work/ops.py", line 12, in step\n    {code}\n'
    if caret:
        frame+='    '+' '*code.index(caret)+'^'*len(caret)+'\n'
    return 'QUESTION\nDiagnose the failure.\nRECORDED TRACEBACK\n'+frame+error+'\nSOURCE CONTEXT\n'+source

class Checks(unittest.TestCase):
    def test_shape_arithmetic(self):
        self.assertEqual(shape_of("weights.T",{"weights":(6,4,2)}),(2,4,6))
        self.assertEqual(shape_of("b[None]",{"b":(8,11)}),(1,8,11))
        self.assertIsNone(shape_of("b[i]",{"b":(8,11)}))
        self.assertIsNone(shape_of("foo(x)",{}))

    def test_source_bound_axis(self):
        facts=dimension_facts("left @ b[None]",(7,8),{"b":(8,11)})
        self.assertIn("axis 1",facts);self.assertIn("= 8",facts);self.assertIn("axis 0",facts)
        self.assertNotIn("axis 2 = 8",facts)
        self.assertNotIn("Source shape",dimension_facts("left @ b[None]",(7,9),{"b":(8,11)}))

    def test_operand_order(self):
        self.assertEqual(dimension_facts("left @ b + b @ c",(7,8),{}),"")
        self.assertEqual(dimension_facts("a if flag else b @ c",(7,8),{}),"")

    def test_caret_boundary(self):
        code='value = reduce(host.asarray(theta))'
        t=Trace(packet(code,'TracerArrayConversionError: array','',caret='host.asarray(theta)'))
        answer=' '.join(position_facts(t))
        self.assertIn('host.asarray(theta)',answer);self.assertIn('reduce',answer)
        t=Trace(packet(code,'TracerArrayConversionError: array','',caret='value'))
        self.assertFalse(position_facts(t))

    def test_context_and_chain(self):
        p=packet('out = f(stream)', 'TracerArrayConversionError: trace','/work/ops.py\n1  raw = data.sample()\n2  stream = normalize(raw)\n3  unrelated = 777\nSOURCE ATTRIBUTION\n4  stream = leak')
        lines=source_rows(p);found=provenance(lines,{"stream"},[])
        cites=" ".join(x.code for x in found)
        self.assertIn('raw = data.sample()',cites);self.assertIn('stream = normalize(raw)',cites)
        self.assertNotIn('777',cites);self.assertNotIn('leak',cites)

    def test_off_target_abstains(self):
        self.assertEqual(card(packet("x = f(y)","ValueError: unrelated"),"a reviewed relation"),None)
        self.assertEqual(card(packet("x = f(y)","TypeError: unrelated"),"a reviewed relation"),None)

    def test_deferred_lambda_is_not_eager(self):
        t=Trace(packet("r = jit(lambda a: np.asarray(a))(v)", "TracerArrayConversionError: trace", caret="np.asarray(a)"))
        self.assertNotIn("before invoking the enclosing call",' '.join(position_facts(t)))

    def test_import_package_is_not_submodule_rebind(self):
        p=packet("x=jax.grad(g)", "TracerArrayConversionError: trace", "/work/ops.py\n1  import jax\n2  import jax.debug")
        t=Trace(p)
        self.assertFalse(binding_facts(source_rows(p),list(t.frames_before_error())))

    def test_no_cross_file_parameter_mixup(self):
        p='RECORDED TRACEBACK\n  File "/root/start.py", line 8, in outer\n    obj.fit(batch)\n  File "/root/missing.py", line 7, in fit\n    return np.array(z)\nTracerArrayConversionError: traced\nSOURCE CONTEXT\n/root/other.py\n1  def fit(self, wrong):'
        self.assertFalse(direct_bindings(source_rows(p),Trace(p)))

    def test_return_source_within_file(self):
        p=packet("r = slab[t]","TracerArrayConversionError: trace","/work/ops.py\n1  def assemble(samples):\n2      return np.asarray(samples)\n3  slab = assemble(seed)")
        fact=" ".join(producer_paths(source_rows(p),{"slab"}))
        self.assertIn("assemble(seed)",fact)
        self.assertIn("np.asarray(samples)",fact)

    def test_composed_stages(self):
        from jaxalyzer.evidence import composition_facts
        p=packet("return np.dot(x,w)","TypeError: trace","/app/layer.py\n12  tl.Serial(enc, tl.Fn('pool', lambda u: u[:, 1, :]), tl.Dense(9))")
        f=' '.join(composition_facts(source_rows(p),Trace(p)))
        self.assertIn("axis 1",f);self.assertIn("tl.Dense(9)",f)

    def test_outer_result_is_pending(self):
        from jaxalyzer.evidence import composition_facts
        p='RECORDED TRACEBACK\n  File "/app/top.py", line 2, in outer\n    act, state = policy.choose(input)\n  File "/app/policy.py", line 9, in choose\n    x = np.array(act)\nTracerArrayConversionError: array\nSOURCE CONTEXT'
        self.assertIn("not returned",' '.join(composition_facts(source_rows(p),Trace(p))))

    def test_starred_tuple_with_adjacent_binding(self):
        p="RECORDED TRACEBACK\n/work/lib.py in train(x)\n  7  args = (values[i], features[i])\n--> 8  loss(*args)\n/work/ops.py in loss(v, f)\n--> 6  return np.matmul(f,v)\nTypeError: error\nSOURCE CONTEXT"
        self.assertIn("`v` ← `values[i]`"," ".join(direct_bindings([],Trace(p))))
        self.assertFalse(direct_bindings([],Trace(p.replace("  7  args", "  4  args"))))
