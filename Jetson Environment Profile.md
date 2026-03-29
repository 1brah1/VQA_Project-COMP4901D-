Jetson Environment Profile: VQA Project
System Core

    Architecture: aarch64 (ARM64)

    JetPack/L4T Version: R35.5.0 (Linux for Tegra)

    Python Version: 3.8.10

Critical AI Stack

    PyTorch: 2.0.0+nv23.05 (CUDA Enabled: Yes)

    Torchvision: [Manual Source Build v0.16.1 detected in user logs]

    Transformers: 4.46.3

    BitsAndBytes: 0.42.0 (Source Compiled: No - Standard pip wheel missing ARM64 CUDA binaries)

Hardware Constraints

    Device: Jetson AGX / Xavier Series (t186ref family)

    CUDA Capability: 8.7

    Shared Memory: Integrated CPU/GPU VRAM (Unified Memory Architecture)

Known Compatibility Issues

    AWQ Kernels: Standard autoawq is incompatible with Python 3.8/Torch 2.0.0 on this architecture. Currently using a fallback loader (Emergency FP16) which causes 0% accuracy/gibberish output.

    Protocols: Requiring manual pathing for LD_LIBRARY_PATH to locate libcudart.so (currently found in /usr/local/cuda/lib64/).

    Binary Mismatch: bitsandbytes requires manual compilation from source for CUDA 11.4 to function on this board.