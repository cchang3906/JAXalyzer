import unittest
from jaxalyzer import emit

class TraceAttribution(unittest.TestCase):
    def test_new_packet_without_case_registry(self):
        from jaxalyzer import analyze_packet
        flat='RECORDED TRACEBACK\n  File "/unseen/new.py", line 9, in metric\n    value = int(amount.max())\n            ^^^^^^^^^^^^^^^^^\nConcretizationTypeError: concrete value required\nSOURCE CONTEXT\n1  import jax.numpy as jnp'
        r=analyze_packet(flat)
        self.assertIsNotNone(r);self.assertIn('amount.max()',r["evidence"])
        self.assertIsNone(analyze_packet(flat.replace("ConcretizationTypeError: concrete value required","ValueError: unrelated")))

    def test_unlabelled_second_site(self):
        for extra in ("", "/tmp/notebook.ipynb セル 7 line 9\n"):
            flat="RECORDED TRACEBACK\n/tmp/main.py in myfn(x)\n--> 2 return third(x)\n"+extra+"--> 9 return np.asarray(x)\nTracerArrayConversionError: tracer\nSOURCE CONTEXT\n1  import numpy as np"
            self.assertEqual(emit(flat,"Approved parent hint."),("parent","Approved parent hint."))

    def test_two_separate_named_sites(self):
        flat="RECORDED TRACEBACK\n/tmp/main.py in myfn(x)\n--> 2 return third(x)\n/tmp/lib.py in third(y)\n--> 9 return np.asarray(y)\nTracerArrayConversionError: tracer\nSOURCE CONTEXT\n1  import numpy as np"
        self.assertEqual(emit(flat,"Approved parent hint.")[0],"expanded")
