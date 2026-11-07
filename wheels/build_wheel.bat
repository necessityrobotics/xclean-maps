@echo off
rem gsplat 1.4.0 compiled wheel for Windows, Python 3.12, torch 2.9.1+cu130. Built by Necessity Robotics for xclean-maps.
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3
set PATH=%CUDA_HOME%\bin;%PATH%
set TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;8.9;9.0;12.0+PTX
set MAX_JOBS=12
set DISTUTILS_USE_SDK=1
set CL=/Zc:preprocessor
set CUDA_VISIBLE_DEVICES=-1
cd /d D:\xrc_work\gsplat_build\b
D:\envs\gsplat_build312\python.exe -m pip wheel . --no-build-isolation --no-deps -w D:\xrc_work\gsplat_build\dist -v
