# xllm x_attention reference (built from source)

The correctness/timing reference is the upstream xLLM Ascend C operator
`aclnnXAttention`. This repo ships **source only**; the build artifacts are not
committed -- run `setup_ref.sh` to build them for your machine (CANN 9.1 /
ascend910_93 by default).

> This `ref/` (i.e. `backends/xllm/ref/`) is the *operator*; the sibling
> `backends/xllm/backend.py` is the thin *backend* (plugin) that calls it via
> `import custom_ops`. Two layers, not duplicates.

```
backends/xllm/ref/
  src/pybind/      OUR bridge (tracked): trimmed RegisterOps.cpp + setup.py
                   (pytorch_npu_helper.hpp is included from the submodule, not vendored)
  python/          custom_ops.py (OUR trimmed wrapper, tracked) + custom_ops_lib*.so (BUILT, gitignored)
  third_party/
    xllm_ops/      git submodule: op source + CANN custom-op build framework (build-time)
  setup_ref.sh     build op (from submodule) + pybind (from src) -> opp/ and python/*.so
  opp/             BUILT (gitignored): installed custom_xllm_math opp vendor package
```

The operator source is **not vendored** -- it lives in the pinned `third_party/
xllm_ops` submodule (single source of truth). The original research repo carried a copy of
`x_attention/`, `common/kernel_common.hpp` and a `catlass_include/` fork; those
were byte-identical distillations of the submodule, so they were dropped. We keep
only what is genuinely ours: the trimmed pybind bridge and `custom_ops.py`.

At **runtime**, `env.sh` points `ASCEND_CUSTOM_OPP_PATH` at `opp/` and
`PYTHONPATH` at `python/`, so once built it needs only CANN + torch_npu (no
dependency on the build framework). The **build framework** is only needed to
(re)build the operator, and is vendored as the `third_party/xllm_ops` submodule.

## Build

```bash
git submodule update --init                   # once, if not cloned already
bash backends/xllm/ref/setup_ref.sh          # SOC=ascend910_93 by default
```

`setup_ref.sh` auto-inits the submodule if missing, plus **only** the nested
`catlass` submodule (the op includes its headers from `third_party/catlass/include`);
`pto-isa` is unused by x_attention and deliberately left uninitialized to avoid
extra network. It then builds the submodule's own `x_attention` (no staging) and
the **pybind** bridge (compiled against torch_npu + CANN; its one upstream header,
`pytorch_npu_helper.hpp`, is included from the submodule rather than vendored).
The framework path is intentionally fixed to this in-repo submodule, so builds
cannot silently use an external `xllm_ops` checkout.

**Build time.** In the original reference environment, the framework compiling
**protobuf + abseil from source** took ~4 min once; the op itself + packaging took
~15s. These are historical measurements, not fresh-checkout validation for this
public repo. `setup_ref.sh` reuses `build/` across runs. Set
`SETUP_REF_CLEAN=1` to force a clean rebuild (e.g. after changing `--soc`). The first
build also needs network to fetch those deps (from gitcode).

A rebuild is only needed when the op source (submodule pin) changes, the CANN version
changes, or you move to a different arch / SoC / Python version (the built artifacts
are `aarch64` + `ascend910_93` + `cpython-312` specific).

## The continuous-vs-paged detail

The operator requires exactly one paged side, so the harness runs **continuous
shared KV + paged unshared KV with an identity request->slot table** -- which is
numerically identical to a continuous unshared layout. See
`backends/xllm/backend.py:adapt`.
