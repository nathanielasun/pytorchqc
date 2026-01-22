# MPS Accelerated Quantum Circuit Simulator

A high-performance statevector quantum circuit simulator with GPU acceleration support for CUDA and Apple Silicon (MPS) via PyTorch.

## Features

- **GPU Acceleration**: Supports NVIDIA CUDA and Apple Silicon MPS backends
- **Efficient Gate Application**: Strided indexing avoids large tensor products
- **Mid-Circuit Measurements**: Full support for measurement and collapse during circuit execution
- **Comprehensive Gate Set**: Clifford gates, rotation gates, controlled gates, and custom unitaries
- **Fluent API**: Method chaining for clean circuit construction
- **Analysis Tools**: Fidelity calculations, statevector inspection, and visualization

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+ (with CUDA or MPS support for GPU acceleration)
- NumPy

### Install Dependencies

```bash
pip install torch numpy
```

For visualization support (optional):
```bash
pip install matplotlib
```

### Package Structure

```
mps_accel_qc/
├── __init__.py    # Package exports and documentation
├── gates.py       # Gate class with quantum gate definitions
├── circuit.py     # Circuit class for building and executing circuits
└── utils.py       # Utility functions for I/O, memory, and analysis
```

## Quick Start

```python
from mps_accel_qc import Circuit

# Create a 2-qubit Bell state
qc = Circuit(2, device='gpu')  # Auto-detects CUDA or MPS
qc.h(0)           # Hadamard on qubit 0
qc.cnot(0, 1)     # CNOT: control=0, target=1

# Execute and measure
qc.execute(shots=1000)
print(qc.get_counts())  # {'00': ~500, '11': ~500}
```

## API Reference

### Circuit Class

The main class for building and executing quantum circuits.

#### Constructor

```python
Circuit(size: int, device: str = None, threads: int = 8)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `size` | `int` | Number of qubits in the circuit |
| `device` | `str` | `'gpu'` (auto-detect), `'cpu'`, `'cuda'`, or `'mps'` |
| `threads` | `int` | CPU threads (only used when `device='cpu'`) |

#### Adding Gates

**General Method:**
```python
qc.add(gate: str, target: int, control: int = None, controls: List[int] = None,
       angle: float = None, angles: List[float] = None)
```

**Shorthand Methods:**

| Method | Description | Example |
|--------|-------------|---------|
| `h(target)` | Hadamard gate | `qc.h(0)` |
| `x(target)` | Pauli-X (NOT) | `qc.x(0)` |
| `y(target)` | Pauli-Y | `qc.y(0)` |
| `z(target)` | Pauli-Z | `qc.z(0)` |
| `s(target)` | S gate (√Z) | `qc.s(0)` |
| `t(target)` | T gate (√S) | `qc.t(0)` |
| `rx(target, angle)` | X-rotation | `qc.rx(0, math.pi/2)` |
| `ry(target, angle)` | Y-rotation | `qc.ry(0, math.pi/4)` |
| `rz(target, angle)` | Z-rotation | `qc.rz(0, math.pi)` |
| `p(target, angle)` | Phase gate | `qc.p(0, math.pi/4)` |
| `u(target, θ, φ, λ)` | Universal U3 | `qc.u(0, pi/2, 0, pi)` |
| `cnot(ctrl, tgt)` | CNOT gate | `qc.cnot(0, 1)` |
| `cx(ctrl, tgt)` | CNOT (alias) | `qc.cx(0, 1)` |
| `cz(ctrl, tgt)` | Controlled-Z | `qc.cz(0, 1)` |
| `swap(q1, q2)` | SWAP gate | `qc.swap(0, 1)` |
| `measure(target)` | Mid-circuit measurement | `qc.measure(0)` |

#### Execution

```python
qc.execute(shots: int = 1024, cache: bool = False)
```

| Parameter | Description |
|-----------|-------------|
| `shots` | Number of measurement samples |
| `cache` | Cache statevector before first MCM (optimization for MCM circuits) |

#### Results

| Method | Returns | Description |
|--------|---------|-------------|
| `get_counts()` | `Dict[str, int]` | Measurement counts (little-endian bitstrings) |
| `get_counts_big_endian()` | `Dict[str, int]` | Measurement counts (big-endian bitstrings) |
| `get_statevector()` | `Tensor` | Final statevector |
| `get_probabilities()` | `Tensor` | Probability distribution |

#### Circuit Information

| Method | Returns | Description |
|--------|---------|-------------|
| `depth()` | `int` | Circuit depth (number of layers) |
| `gate_count()` | `Dict[str, int]` | Count of each gate type |
| `print_circuit()` | None | Print text representation |
| `len(qc)` | `int` | Total number of gates |

### Gate Class

Low-level gate definitions and application methods.

```python
from mps_accel_qc import Gate
import torch

gate = Gate(device=torch.device('cpu'))
```

#### Available Gates

**Fixed Gates (Tensors):**
- `gate.H` - Hadamard
- `gate.X`, `gate.Y`, `gate.Z` - Pauli gates
- `gate.S`, `gate.Sdg` - S and S-dagger
- `gate.T`, `gate.Tdg` - T and T-dagger
- `gate.SX` - √X gate
- `gate.I` - Identity

**Parametric Gates (Methods):**
- `gate.Rx(angle)` - X-rotation
- `gate.Ry(angle)` - Y-rotation
- `gate.Rz(angle)` - Z-rotation
- `gate.P(phi)` - Phase shift
- `gate.Ph(phase)` - Global phase
- `gate.U(theta, phi, lam)` - Universal U3 gate

**Two-Qubit Gates:**
- `gate.CNOT(state, control, target)` - Controlled-X
- `gate.CZ(state, control, target)` - Controlled-Z
- `gate.SWAP(state, qubit1, qubit2)` - SWAP

**Measurement:**
- `gate.measure(state, target)` - Mid-circuit measurement with collapse

#### Direct Gate Application

```python
# Apply gate directly to statevector
gate.apply(gate_matrix, state, target, controls=None)
```

### Utility Functions

#### Memory Monitoring

```python
from mps_accel_qc import (
    get_gpu_memory_usage,      # Returns MB used (float or None)
    get_gpu_memory_cached,     # Returns MB cached (CUDA only)
    print_memory_stats,        # Prints memory summary
    estimate_statevector_memory,  # Estimate MB for n qubits
)

# Example
print(f"Memory used: {get_gpu_memory_usage():.2f} MB")
print(f"20 qubits needs: {estimate_statevector_memory(20):.2f} MB")
```

#### Device Utilities

```python
from mps_accel_qc import (
    get_available_devices,  # {'cpu': True, 'cuda': False, 'mps': True}
    get_best_device,        # Returns torch.device
    get_device_info,        # Detailed device information
)
```

#### Statevector I/O

```python
from mps_accel_qc import save_statevector, load_statevector

# Save in different formats
save_statevector(qc.states, "state.bin", format='binary')  # Compact
save_statevector(qc.states, "state.npy", format='numpy')   # With metadata
save_statevector(qc.states, "state.pt", format='torch')    # Full tensor info

# Load back
states = load_statevector("state.bin", format='binary', device='cuda')
```

#### Analysis Functions

```python
from mps_accel_qc import (
    fidelity,              # |<ψ1|ψ2>|² between two states
    trace_distance,        # √(1 - fidelity)
    get_nonzero_amplitudes,  # Dict of basis -> amplitude
    print_statevector,     # Human-readable statevector display
    counts_to_probabilities,  # Convert counts to probabilities
    plot_histogram,        # Matplotlib histogram of results
)

# Example
f = fidelity(state_ideal, state_noisy)
print(f"Fidelity: {f:.6f}")

print_statevector(qc.states, threshold=1e-6, max_terms=10)
```

## Examples

### Bell State

```python
from mps_accel_qc import Circuit

qc = Circuit(2, device='gpu')
qc.h(0).cnot(0, 1)
qc.execute(shots=1000)

print(qc.get_counts())
# Output: {'00': 512, '11': 488}
```

### GHZ State (3-qubit entanglement)

```python
qc = Circuit(3)
qc.h(0)
qc.cnot(0, 1)
qc.cnot(1, 2)
qc.execute(shots=1000)

print(qc.get_counts())
# Output: {'000': 495, '111': 505}
```

### Rotation Gates

```python
import math

qc = Circuit(1)
qc.ry(0, math.pi / 4)  # Rotate towards |1⟩
qc.execute(shots=1000)

# ~85% |0⟩, ~15% |1⟩
```

### Controlled Operations

```python
qc = Circuit(3)
qc.h(0)
qc.h(1)
# Multi-controlled X (Toffoli-like)
qc.add('X', target=2, controls=[0, 1])
qc.execute(shots=1000)
```

### Mid-Circuit Measurement

```python
qc = Circuit(2)
qc.h(0)
qc.measure(0)  # Collapses qubit 0
qc.cnot(0, 1)  # Behavior depends on measurement outcome
qc.execute(shots=1000, cache=True)  # cache=True for efficiency
```

### Quantum Teleportation

```python
import math

qc = Circuit(3)

# Prepare state to teleport
qc.rx(0, math.pi / 3)

# Create Bell pair
qc.h(1)
qc.cnot(1, 2)

# Bell measurement
qc.cnot(0, 1)
qc.h(0)

qc.execute(shots=1000)
print(qc.get_counts())
```

### Method Chaining

```python
result = (
    Circuit(2, device='gpu')
    .h(0)
    .cnot(0, 1)
    .rz(0, math.pi/4)
    .execute(shots=1000)
)
print(result.get_counts())
```

## Performance Notes

### Memory Requirements

| Qubits | Amplitudes | Memory (cfloat) |
|--------|------------|-----------------|
| 10 | 1,024 | 8 KB |
| 15 | 32,768 | 256 KB |
| 20 | 1,048,576 | 8 MB |
| 25 | 33,554,432 | 256 MB |
| 30 | 1,073,741,824 | 8 GB |

### Device Selection

- **CUDA**: Best for circuits with 20+ qubits
- **MPS (Apple Silicon)**: Good performance, unified memory architecture
- **CPU**: Use multi-threading for moderate circuits (10-20 qubits)

```python
# Auto-select best available GPU
qc = Circuit(20, device='gpu')

# Force CPU with 8 threads
qc = Circuit(15, device='cpu', threads=8)
```

### Optimization Tips

1. **Reuse circuits**: Call `qc.reset()` instead of creating new circuits
2. **Batch measurements**: Use higher `shots` count instead of multiple executions
3. **MCM caching**: Enable `cache=True` when circuit has substantial work before first measurement
4. **Device transfer**: Minimize transfers between CPU and GPU

## Qubit Indexing Convention

This simulator uses **little-endian** bit ordering:
- Qubit 0 is the **least significant bit** (rightmost)
- The state `|abc⟩` means qubit 2 = a, qubit 1 = b, qubit 0 = c

```python
qc = Circuit(3)
qc.x(0)  # Sets qubit 0 to |1⟩
qc.execute(shots=100)
print(qc.get_counts())  # {'001': 100} (little-endian)
print(qc.get_counts_big_endian())  # {'100': 100}
```

## Gate Definitions

### Single-Qubit Gates

**Hadamard (H):**
```
H = 1/√2 × [[1,  1],
            [1, -1]]
```

**Pauli Gates:**
```
X = [[0, 1],    Y = [[0, -i],    Z = [[1,  0],
     [1, 0]]         [i,  0]]         [0, -1]]
```

**Phase Gates:**
```
S = [[1, 0],    T = [[1,    0   ],    P(φ) = [[1,    0   ],
     [0, i]]         [0, e^(iπ/4)]]           [0, e^(iφ)]]
```

**Rotation Gates:**
```
Rx(θ) = [[cos(θ/2),   -i·sin(θ/2)],
         [-i·sin(θ/2), cos(θ/2)  ]]

Ry(θ) = [[cos(θ/2), -sin(θ/2)],
         [sin(θ/2),  cos(θ/2)]]

Rz(θ) = [[e^(-iθ/2),    0     ],
         [   0,      e^(iθ/2)]]
```

**Universal U3 Gate:**
```
U(θ,φ,λ) = [[cos(θ/2),        -e^(iλ)·sin(θ/2)     ],
            [e^(iφ)·sin(θ/2),  e^(i(φ+λ))·cos(θ/2)]]
```

### Two-Qubit Gates

**CNOT (CX):** Flips target if control is |1⟩
**CZ:** Applies Z to target if control is |1⟩
**SWAP:** Exchanges states of two qubits

## License

MIT License

## Author

Nathaniel Sun
