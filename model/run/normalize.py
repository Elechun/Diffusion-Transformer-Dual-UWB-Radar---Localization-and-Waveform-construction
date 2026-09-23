"""The z-score helper the whole pipeline shares.

It lived in the scoring module, which made every processing file import the scorer just to normalise an
array. The epsilon is 1e-12 and is kept exactly as it was — several published numbers were produced with
it, so this is a move, not a rewrite. (`signal_processing`, `conditioner` and `diffusion_model` each carry
their own 1e-9 variant for their own arrays; those are left alone for the same reason.)
"""
import numpy as np

zn = lambda x: (x - np.mean(x)) / (np.std(x) + 1e-12)
