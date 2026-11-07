# Installing xclean-maps

xclean-maps needs four things:

1. Python 3.12 (the xclean environment, or a new one).
2. PyTorch 2.9.1 for CUDA 13.0.
3. gsplat 1.4.0 (the rasterizer). On Windows, use OUR prebuilt wheel (step 3 below).
4. xclean itself (the sequence reader and the 32-direction SGM; it brings CuPy for CUDA 13).

A real map needs an NVIDIA GPU. The driver must support CUDA 13 (on Windows: driver 580 or newer).
The CPU can run the tests, `xclean-maps info` and `xclean-maps view` (points mode) only.

## Main route (Windows 11, x64)

Use the Anaconda Prompt. Install xclean first, as xclean's README says (`conda env create -f environment.yml`,
`conda activate xclean`, `pip install -e .`). Then work in a COPY of that environment, so xclean's own environment
stays as tested:

```
conda create -n xclean_maps --clone xclean
conda activate xclean_maps
```

Or make a new environment with Python 3.12 and install xclean into it.

Then:

```
pip uninstall -y opencv-python                             # xclean pins opencv-python; OpenCV 5 keeps HOG (people mask) only in contrib
pip install opencv-contrib-python==5.0.0.93                # same version as xclean's pin, a superset (xclean doctor: 24 OK)
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
pip install https://github.com/necessityrobotics/xclean-maps/raw/v0.1.0/wheels/gsplat-1.4.0-cp312-cp312-win_amd64.whl
pip install https://github.com/necessityrobotics/xclean-maps/archive/refs/tags/v0.1.0.zip   # or, from a clone: pip install .
```

Check the install:

```
xclean-maps --help
python -c "import torch, gsplat; print(torch.__version__, gsplat.__version__, torch.cuda.is_available())"
```

The last line must print `2.9.1+cu130 1.4.0 True`.

## Troubleshooting

| Message | Fix |
|---|---|
| `cv2 has no attribute HOGDescriptor` / `this OpenCV ... has no HOGDescriptor` | `pip uninstall -y opencv-python` then `pip install opencv-contrib-python==5.0.0.93` |
| `torch.cuda.is_available()` is False | update the NVIDIA driver (CUDA 13 needs a 580-series or newer driver); check that `torch.__version__` ends in `+cu130` |
| `DLL load failed` when importing gsplat | torch is not 2.9.1+cu130, or Python is not 3.12; reinstall step 2 in that environment |
| out of GPU memory (K5) | use `--preset small`; close other GPU programs |
| the build cannot find the sequence | add `--sequence <the folder you gave xclean>` |
| a stage stopped | run the same command again: finished stages are skipped and training resumes |
| the browser page stays black | wait for "map loaded", then click the page once; use Chrome, Edge or Firefox |

About our gsplat wheel:

- It is an UNMODIFIED build of gsplat 1.4.0 (Apache-2.0, nerfstudio-project), from the gsplat 1.4.0 files on PyPI.
  See NOTICE and wheels/GSPLAT_WHEEL_BUILD_RECORD.md (inputs, hashes, flags, checks). sha256 of the wheel:
  caeddb524e90ef5740b73f0dabcf94aad0cfc5097127c3fc10aa9d1cd11c7b9c (wheels/SHA256SUMS.txt).
- Tested: its renders are bit-identical to a just-in-time gsplat 1.4.0 build (20,000 Gaussians, 640x400).
- Built for: Windows x64, CPython 3.12, PyTorch 2.9.1 + CUDA 13.0.
- GPU architectures in the wheel: 7.5, 8.0, 8.6, 8.9, 9.0, 12.0, plus PTX for newer GPUs.
- It needs exactly torch 2.9.1 (the compiled extension is tied to that PyTorch ABI). With another PyTorch version,
  use the fallback below.
- No compiler and no CUDA Toolkit are needed with the wheel.

## Fallback A: compile gsplat yourself on Windows

Use this only when our wheel does not fit your setup (another Python, another PyTorch).

You need:

- Visual Studio 2022 Build Tools (or newer) with "Desktop development with C++" (MSVC v143, Windows SDK).
- The NVIDIA CUDA Toolkit whose major.minor version matches your PyTorch build (for torch 2.9.1+cu130: CUDA 13.0).
- `ninja` (`pip install ninja`).

Build a wheel once, from the "x64 Native Tools Command Prompt" (it sets up MSVC), in your environment, from the
gsplat 1.4.0 source (`git clone --recursive --branch v1.4.0 https://github.com/nerfstudio-project/gsplat`; the
`--recursive` gets the bundled GLM):

```
set DISTUTILS_USE_SDK=1
set TORCH_CUDA_ARCH_LIST=8.6;8.9;12.0+PTX            # your GPU(s); fewer architectures = faster build
set MAX_JOBS=8
cd gsplat
pip wheel . --no-deps --no-build-isolation -w dist
pip install dist\gsplat-1.4.0-*.whl
```

The build takes a long time (tens of minutes; it depends on the number of architectures). We have not run these exact
commands on a clean machine; the recipe of our release wheel is in the release notes.

Another way is gsplat's just-in-time (JIT) build: `pip install gsplat==1.4.0` installs the Python code only, and the
first map build then compiles the CUDA code. It needs the same MSVC + CUDA Toolkit + ninja, run from a prompt where MSVC
is set up. The compiled code is cached (in `%USERPROFILE%\.cache\torch_extensions`, or in `TORCH_EXTENSIONS_DIR` when
you set it), so it compiles once.

## Fallback B: Linux (UNTESTED)

We have not tested xclean-maps on Linux.

```
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
pip install gsplat==1.4.0          # compiles the CUDA code (JIT, at first use) - needs the CUDA Toolkit and a C++ compiler
pip install xclean-maps
```

xclean itself must be installed as its README says.

## Tests (CPU, no GPU needed)

From a source tree:

```
set CUDA_VISIBLE_DEVICES=-1
python -I -B -m unittest discover -s tests -v
```

Tests that need our own recordings skip when the data is not there (see the top of tests/test_cpu.py).
