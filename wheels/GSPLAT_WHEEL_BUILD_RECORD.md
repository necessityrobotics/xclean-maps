# gsplat 1.4.0 Windows wheel for xclean-maps (built 2026-11-08 02:40-02:49)

File: `dist/gsplat-1.4.0-cp312-cp312-win_amd64.whl` (17100687 bytes), sha256 `caeddb524e90ef5740b73f0dabcf94aad0cfc5097127c3fc10aa9d1cd11c7b9c`.
Unmodified gsplat 1.4.0 (Apache-2.0, nerfstudio-project; bundles glm, MIT). Built by Necessity Robotics.

## Inputs
- Python code + CUDA sources (with glm): PyPI `gsplat-1.4.0-py3-none-any.whl`, sha256 283d96724a8e41af4b338a2882285845ef2e04f3abcdb158bfbe9d515a64cdf0 (identical to the gsplat that made our website maps in cvcuda133).
- setup.py / setup.cfg / LICENSE: PyPI sdist `gsplat-1.4.0.tar.gz`, sha256 8aa81a785e0daf3ed60d0b9930a56c0f337280e6989351d1f1b74e21cf190160. The sdist lacks third_party/glm; its rendering.py differs from the wheel's only in 2 docstring lines.

## Toolchain
Python 3.12.16, torch 2.9.1+cu130, CUDA toolkit 13.3 (nvcc V13.3.73), MSVC Build Tools 18, ninja. Env D:/envs/gsplat_build312.
Flags: TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;8.9;9.0;12.0+PTX, MAX_JOBS=12, CL=/Zc:preprocessor (CUDA 13.3 CCCL needs the conforming MSVC preprocessor; source unchanged). Script: build_wheel.bat.

## Checks
- cuobjdump: SASS for sm_75, sm_80, sm_86, sm_89, sm_90, sm_120 (24 kernels each) + PTX compute_120.
- Install into a venv (D:/envs/gsplat_wheeltest): gsplat loads the compiled `gsplat/csrc.pyd` (no JIT compile).
- GPU parity vs the proven JIT build (cvcuda133), 20,000 random Gaussians, 640x400, one forward + backward (parity_render.py):
  image max|d| = 0, alpha max|d| = 0 (bit-identical); means gradient max|d| = 3.7e-4, mean |d| = 1.8e-6 against mean |g| = 24.5 (atomic-add order).
## Limits
- Needs torch 2.9.x built for CUDA 13 (cu130) and Python 3.12 on Windows x64. NVIDIA driver must support CUDA 13.
- The test venv reused the build env's torch (--system-site-packages). A fully fresh install test is still to do (other device).
