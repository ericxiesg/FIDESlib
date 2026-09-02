# pyfideslib

Python bindings for the fideslib API. One context serves both backends:

```python
import pyfideslib as pf
e = pf.Engine("cpu")      # OpenFHE
e = pf.Engine("cuda:0")   # FIDESlib CUDA
ct = e.encrypt(np.random.rand(e.slots)); print(e.decrypt_real(e.rotate(ct, 3)))
```

Build (after installing fideslib):

    cmake -S python -B build-py -Dfideslib_DIR=<install>/lib/cmake/fideslib && cmake --build build-py -j
    export PYTHONPATH=$PWD/python
    PYFIDESLIB_DEVICES=cpu       pytest python/tests -x -v      # stage 1..4 on OpenFHE
    PYFIDESLIB_DEVICES=cuda:0    pytest python/tests -x -v      # same on the GPU
    PYFIDESLIB_DEVICES=cpu,cuda:0 pytest python/tests -k stage3  # only lazy relin

Stages: 1 I/O round trip · 2 linear ops + level-aware rotation keys · 3 lazy relinearisation ·
4 bootstrap with output-level contract (`PYFIDESLIB_BOOT_LOGN/DEPTH` to size it).
`Engine.multiply(ct, ct)` is lazy (degree-2) by default; call `relinearize` once per accumulation.
