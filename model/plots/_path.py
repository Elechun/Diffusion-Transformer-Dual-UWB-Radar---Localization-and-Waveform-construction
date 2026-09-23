"""Makes the three sibling groups importable however a script is launched.

The modules are split by role - run/ operates the model, eval/ scores it, plots/ draws it - but they are
plain scripts, not a package, so `import network` has to keep working from any of the three. Every module
starts with `import _path` to arrange that. Each directory holds an identical copy.
"""
import os,sys
_d=os.path.dirname(os.path.abspath(__file__))
for _g in ('run','eval','plots'):
    _p=os.path.join(os.path.dirname(_d),_g)
    if _p not in sys.path: sys.path.insert(0,_p)
