@echo off
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
set "CUDA_PATH_V11_8=%CUDA_PATH%"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"
set "INCLUDE=%CUDA_PATH%\include;%INCLUDE%"
set "LIB=%CUDA_PATH%\lib\x64;%LIB%"
