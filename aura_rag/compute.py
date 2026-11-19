"""Hardware detection for local model inference."""


def available_gpu():
    """Return a usable PyTorch GPU as (device, display_name), or None.

    CUDA also covers PyTorch's ROCm builds. XPU and Apple MPS are checked as
    secondary backends so the same preference works on more than NVIDIA PCs.
    """
    try:
        import torch
    except Exception:
        return None

    try:
        if torch.cuda.is_available():
            return "cuda:0", torch.cuda.get_device_name(0)
    except Exception:
        pass

    try:
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            return "xpu:0", xpu.get_device_name(0)
    except Exception:
        pass

    try:
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "mps", "Apple Metal GPU"
    except Exception:
        pass

    return None


def llama_gpu_offload_available():
    """Return True/False when llama.cpp reports a GPU backend, else None."""
    try:
        import llama_cpp
    except Exception:
        return None

    for module in (llama_cpp, getattr(llama_cpp, "llama_cpp", None)):
        check = getattr(module, "llama_supports_gpu_offload", None)
        if callable(check):
            try:
                return bool(check())
            except Exception:
                return None
    return None
