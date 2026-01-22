"""
MPS Accelerated Quantum Circuit Simulator.

A high-performance statevector quantum circuit simulator supporting
CPU, CUDA, and Apple Silicon (MPS) backends via PyTorch.

Features:
- Efficient strided gate application (no large tensor products)
- Support for mid-circuit measurements
- Parametric gates (Rx, Ry, Rz, U, P)
- Controlled gates (CNOT, CZ, multi-controlled)
- GPU acceleration (CUDA and Apple Silicon MPS)

Basic Usage:
    >>> from mps_accel_qc import Circuit
    >>>
    >>> # Create a 2-qubit circuit
    >>> qc = Circuit(2, device='gpu')
    >>>
    >>> # Build a Bell state
    >>> qc.h(0)           # Hadamard on qubit 0
    >>> qc.cnot(0, 1)     # CNOT with control=0, target=1
    >>>
    >>> # Execute and get results
    >>> qc.execute(shots=1000)
    >>> print(qc.get_counts())  # ~50% |00>, ~50% |11>

Advanced Usage:
    >>> from mps_accel_qc import Circuit, Gate
    >>> import math
    >>>
    >>> # Create circuit with rotation gates
    >>> qc = Circuit(3, device='cpu', threads=4)
    >>> qc.rx(0, math.pi/4)
    >>> qc.ry(1, math.pi/2)
    >>> qc.u(2, math.pi/3, math.pi/4, math.pi/6)
    >>>
    >>> # Add controlled operations
    >>> qc.add('X', target=2, controls=[0, 1])  # Toffoli-like
    >>>
    >>> # Execute and analyze
    >>> qc.execute(shots=10000)
    >>> print(qc.get_statevector())
    >>> print(qc.get_probabilities())

Author: Nathaniel Sun
"""

__version__ = "0.1.0"
__author__ = "Nathaniel Sun"

# Core classes
from .gates import Gate
from .circuit import Circuit

# Utility functions
from .utils import (
    # Memory monitoring
    get_gpu_memory_usage,
    get_gpu_memory_cached,
    print_memory_stats,
    estimate_statevector_memory,

    # I/O
    save_statevector,
    load_statevector,

    # Device utilities
    get_available_devices,
    get_best_device,
    get_device_info,

    # Analysis
    fidelity,
    trace_distance,
    get_nonzero_amplitudes,
    print_statevector,

    # Measurement utilities
    counts_to_probabilities,
    plot_histogram,
)

# Public API
__all__ = [
    # Version info
    "__version__",
    "__author__",

    # Core classes
    "Gate",
    "Circuit",

    # Memory utilities
    "get_gpu_memory_usage",
    "get_gpu_memory_cached",
    "print_memory_stats",
    "estimate_statevector_memory",

    # I/O utilities
    "save_statevector",
    "load_statevector",

    # Device utilities
    "get_available_devices",
    "get_best_device",
    "get_device_info",

    # Analysis utilities
    "fidelity",
    "trace_distance",
    "get_nonzero_amplitudes",
    "print_statevector",

    # Measurement utilities
    "counts_to_probabilities",
    "plot_histogram",
]
