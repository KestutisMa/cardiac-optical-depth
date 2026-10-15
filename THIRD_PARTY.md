# Third-party dependencies

No openCARP or PyXOpto implementation, third-party source tree, or third-party
model weights are distributed here. The electrophysiology and photon-transport
simulators are not part of this analysis release. In particular, the PolyForm
licence does not apply to PyXOpto (GPL-3.0) or to openCARP.

Install dependencies from their upstream distributions; their licences remain
applicable independently of this repository:

| Dependency | Upstream | Licence |
| --- | --- | --- |
| NumPy | https://numpy.org/ | BSD-3-Clause |
| SciPy | https://scipy.org/ | BSD-3-Clause |
| PyTorch | https://pytorch.org/ | BSD-style, with upstream third-party notices |
| PyYAML | https://pyyaml.org/ | MIT |
| Matplotlib | https://matplotlib.org/ | Matplotlib licence (PSF-based) |
| h5py (optional extraction) | https://www.h5py.org/ | BSD-3-Clause |
| EMD-signal / PyEMD (optional denoising) | https://github.com/laszukdawid/PyEMD | Apache-2.0 |

Upstream package distributions contain the authoritative licence texts and
notices, including those of their transitive dependencies. They are not
relicensed under this repository's noncommercial terms.
