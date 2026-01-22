"""
Utility Functions for Quantum Circuit Simulation.

This module provides utility functions for:
- GPU memory monitoring
- Statevector I/O (save/load)
- Device detection and configuration
- Visualization helpers

Author: Nathaniel Sun
"""

import math
from typing import Optional, Dict, Any, Union
from pathlib import Path

import torch
from torch import Tensor
import numpy as np


# =============================================================================
# Memory Monitoring
# =============================================================================

def get_gpu_memory_usage() -> Optional[float]:
    """
    Get current GPU memory usage in megabytes.

    Works with both CUDA and Apple Silicon MPS backends.

    Returns:
        Memory usage in MB, or None if no GPU is available.

    Example:
        >>> mem = get_gpu_memory_usage()
        >>> if mem is not None:
        ...     print(f"GPU memory: {mem:.2f} MB")
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    elif torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / (1024 ** 2)
    return None


def get_gpu_memory_cached() -> Optional[float]:
    """
    Get cached (reserved) GPU memory in megabytes.

    This includes memory that PyTorch has allocated but isn't currently
    using for tensors (kept for future allocations).

    Returns:
        Cached memory in MB, or None if not available.
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / (1024 ** 2)
    return None


def print_memory_stats() -> None:
    """Print a summary of current memory usage."""
    used = get_gpu_memory_usage()
    cached = get_gpu_memory_cached()

    if used is not None:
        device_name = "CUDA" if torch.cuda.is_available() else "MPS"
        print(f"[{device_name}] Memory allocated: {used:.2f} MB")
        if cached is not None:
            print(f"[{device_name}] Memory cached: {cached:.2f} MB")
    else:
        print("[CPU] GPU memory stats not available")


def estimate_statevector_memory(num_qubits: int, dtype: torch.dtype = torch.cfloat) -> float:
    """
    Estimate memory required for a statevector.

    Args:
        num_qubits: Number of qubits.
        dtype: Data type of the statevector elements.

    Returns:
        Estimated memory in megabytes.

    Example:
        >>> mem = estimate_statevector_memory(20)
        >>> print(f"20 qubits needs ~{mem:.2f} MB")
    """
    num_amplitudes = 2 ** num_qubits

    # Get element size based on dtype
    if dtype == torch.cfloat:
        bytes_per_element = 8  # 2 x float32
    elif dtype == torch.cdouble:
        bytes_per_element = 16  # 2 x float64
    elif dtype == torch.float32:
        bytes_per_element = 4
    elif dtype == torch.float64:
        bytes_per_element = 8
    else:
        bytes_per_element = 8  # Default assumption

    total_bytes = num_amplitudes * bytes_per_element
    return total_bytes / (1024 ** 2)


# =============================================================================
# Statevector I/O
# =============================================================================

def save_statevector(
    states: Tensor,
    filename: Union[str, Path],
    format: str = 'binary'
) -> None:
    """
    Save a statevector to a file.

    Args:
        states: The statevector tensor to save.
        filename: Output filename.
        format: File format - 'binary' (raw), 'numpy' (.npy), or 'torch' (.pt).

    Example:
        >>> save_statevector(qc.states, "my_state.bin")
        >>> save_statevector(qc.states, "my_state.npy", format='numpy')
    """
    filename = Path(filename)

    if format == 'binary':
        # Save as raw binary (most compact, requires knowing dtype to load)
        states.cpu().numpy().tofile(filename)
        print(f"Saved statevector to {filename} (binary format)")

    elif format == 'numpy':
        # Save as numpy array (includes dtype metadata)
        np.save(filename, states.cpu().numpy())
        print(f"Saved statevector to {filename} (numpy format)")

    elif format == 'torch':
        # Save as torch tensor (preserves full tensor metadata)
        torch.save(states.cpu(), filename)
        print(f"Saved statevector to {filename} (torch format)")

    else:
        raise ValueError(f"Unknown format '{format}'. Use 'binary', 'numpy', or 'torch'.")


def load_statevector(
    filename: Union[str, Path],
    format: str = 'binary',
    dtype: torch.dtype = torch.cfloat,
    device: Union[str, torch.device] = 'cpu'
) -> Tensor:
    """
    Load a statevector from a file.

    Args:
        filename: Input filename.
        format: File format - 'binary', 'numpy', or 'torch'.
        dtype: Data type (only needed for binary format).
        device: Device to load the tensor onto.

    Returns:
        The loaded statevector tensor.

    Example:
        >>> states = load_statevector("my_state.bin", device='cuda')
        >>> states = load_statevector("my_state.npy", format='numpy')
    """
    filename = Path(filename)

    if isinstance(device, str):
        device = torch.device(device)

    if format == 'binary':
        with open(filename, 'rb') as f:
            buf = f.read()
        tensor = torch.frombuffer(bytearray(buf), dtype=dtype)
        return tensor.clone().to(device)

    elif format == 'numpy':
        arr = np.load(filename)
        return torch.from_numpy(arr).to(device)

    elif format == 'torch':
        return torch.load(filename, map_location=device)

    else:
        raise ValueError(f"Unknown format '{format}'. Use 'binary', 'numpy', or 'torch'.")


# =============================================================================
# Device Utilities
# =============================================================================

def get_available_devices() -> Dict[str, bool]:
    """
    Check which computation devices are available.

    Returns:
        Dictionary indicating availability of each device type.

    Example:
        >>> devices = get_available_devices()
        >>> print(devices)
        {'cpu': True, 'cuda': False, 'mps': True}
    """
    return {
        'cpu': True,
        'cuda': torch.cuda.is_available(),
        'mps': torch.backends.mps.is_available(),
    }


def get_best_device() -> torch.device:
    """
    Get the best available computation device.

    Priority: CUDA > MPS > CPU

    Returns:
        The best available torch.device.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_device_info() -> Dict[str, Any]:
    """
    Get detailed information about available devices.

    Returns:
        Dictionary with device information.
    """
    info: Dict[str, Any] = {
        'best_device': str(get_best_device()),
        'available': get_available_devices(),
    }

    if torch.cuda.is_available():
        info['cuda'] = {
            'device_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device(),
            'device_name': torch.cuda.get_device_name(),
            'memory_total_mb': torch.cuda.get_device_properties(0).total_memory / (1024**2),
        }

    return info


# =============================================================================
# Statevector Analysis
# =============================================================================

def fidelity(state1: Tensor, state2: Tensor) -> float:
    """
    Calculate the fidelity between two pure states.

    Fidelity F = |<psi1|psi2>|^2

    Args:
        state1: First statevector.
        state2: Second statevector.

    Returns:
        Fidelity value between 0 and 1.

    Example:
        >>> f = fidelity(state_ideal, state_noisy)
        >>> print(f"Fidelity: {f:.6f}")
    """
    if state1.shape != state2.shape:
        raise ValueError("Statevectors must have the same shape")

    # Move to same device if needed
    if state1.device != state2.device:
        state2 = state2.to(state1.device)

    inner_product = torch.vdot(state1.flatten(), state2.flatten())
    return abs(inner_product.item()) ** 2


def trace_distance(state1: Tensor, state2: Tensor) -> float:
    """
    Calculate the trace distance between two pure states.

    For pure states: D = sqrt(1 - |<psi1|psi2>|^2)

    Args:
        state1: First statevector.
        state2: Second statevector.

    Returns:
        Trace distance value between 0 and 1.
    """
    f = fidelity(state1, state2)
    return math.sqrt(1 - f)


def get_nonzero_amplitudes(
    states: Tensor,
    threshold: float = 1e-10,
    num_qubits: Optional[int] = None
) -> Dict[str, complex]:
    """
    Get dictionary of non-zero amplitudes in a statevector.

    Args:
        states: The statevector.
        threshold: Minimum magnitude to consider non-zero.
        num_qubits: Number of qubits (inferred if not provided).

    Returns:
        Dictionary mapping bitstrings to complex amplitudes.

    Example:
        >>> amps = get_nonzero_amplitudes(qc.states)
        >>> for basis, amp in amps.items():
        ...     print(f"|{basis}> : {amp}")
    """
    if num_qubits is None:
        num_qubits = int(math.log2(states.numel()))

    result = {}
    states_cpu = states.cpu()

    for i, amp in enumerate(states_cpu):
        if abs(amp) > threshold:
            bitstring = format(i, f'0{num_qubits}b')
            result[bitstring] = complex(amp)

    return result


def print_statevector(
    states: Tensor,
    threshold: float = 1e-10,
    max_terms: int = 20,
    precision: int = 4
) -> None:
    """
    Print a human-readable representation of the statevector.

    Args:
        states: The statevector to display.
        threshold: Minimum magnitude to display.
        max_terms: Maximum number of terms to show.
        precision: Decimal places for amplitude display.
    """
    num_qubits = int(math.log2(states.numel()))
    amps = get_nonzero_amplitudes(states, threshold, num_qubits)

    # Sort by magnitude (descending)
    sorted_amps = sorted(amps.items(), key=lambda x: abs(x[1]), reverse=True)

    print(f"Statevector ({num_qubits} qubits, {len(amps)} non-zero terms):")
    print("-" * 40)

    for i, (basis, amp) in enumerate(sorted_amps[:max_terms]):
        real = amp.real
        imag = amp.imag
        prob = abs(amp) ** 2

        if abs(imag) < 1e-10:
            amp_str = f"{real:+.{precision}f}"
        elif abs(real) < 1e-10:
            amp_str = f"{imag:+.{precision}f}i"
        else:
            amp_str = f"({real:+.{precision}f}{imag:+.{precision}f}i)"

        print(f"  |{basis}> : {amp_str}  (p={prob:.{precision}f})")

    if len(sorted_amps) > max_terms:
        print(f"  ... and {len(sorted_amps) - max_terms} more terms")

    print("-" * 40)


# =============================================================================
# Measurement Utilities
# =============================================================================

def counts_to_probabilities(counts: Dict[str, int]) -> Dict[str, float]:
    """
    Convert measurement counts to probabilities.

    Args:
        counts: Dictionary of measurement counts.

    Returns:
        Dictionary of probabilities.
    """
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def plot_histogram(
    counts: Dict[str, int],
    title: str = "Measurement Results",
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot a histogram of measurement results.

    Requires matplotlib to be installed.

    Args:
        counts: Dictionary of measurement counts.
        title: Plot title.
        figsize: Figure size as (width, height).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting. Install with: pip install matplotlib")
        return

    # Sort by bitstring
    sorted_items = sorted(counts.items())
    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(labels, values)
    ax.set_xlabel('Basis State')
    ax.set_ylabel('Counts')
    ax.set_title(title)

    # Rotate labels if there are many states
    if len(labels) > 8:
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()
