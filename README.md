
# RayTracer

A high-performance ray tracing engine built with CUDA for real-time rendering.
This project supports reflection, refraction, and Fresnel effects, making it capable of simulating realistic materials like glass and mirrors. It demonstrates physically-based rendering concepts, optimized GPU acceleration, and interactive scene visualization.

## Installation

You will need CUDA toolkit, install from here https://developer.nvidia.com/cuda-downloads

```bash
  git clone https://github.com/HellBats/RayTracer
  cd RayTracer 
```
    
## Build
```bash
    mkdir build && cd build
    cmake ..
    make
```