"""
Quantum Circuit Construction and Execution.

This module provides the Circuit class for building quantum circuits,
executing them via statevector simulation, and sampling measurement outcomes.
Supports CPU, CUDA, and Apple Silicon (MPS) backends via PyTorch.

Author: Nathaniel Sun
"""

import math
from typing import Optional, List, Dict, Any, Union

import torch
from torch import Tensor

from gates import Gate


class Circuit:
    """
    A quantum circuit for statevector simulation.

    This class provides methods for constructing quantum circuits by adding
    gates, executing the circuit to obtain the final statevector, and
    sampling measurement outcomes.

    Attributes:
        size: Number of qubits in the circuit.
        device: PyTorch device used for computation.
        states: The statevector after circuit execution.
        measurements: Measurement counts after execution.
        circuit: List of gate operations in the circuit.

    Example:
        >>> qc = Circuit(2, device='cpu')
        >>> qc.add('H', target=0)
        >>> qc.add('CNOT', control=0, target=1)
        >>> qc.execute(shots=1000)
        >>> print(qc.get_counts())  # Bell state: ~50% |00>, ~50% |11>
    """

    # Gate name aliases for user convenience
    GATE_ALIASES = {
        'NOT': 'X',
        'PHASE': 'P',
        'RX': 'Rx',
        'RY': 'Ry',
        'RZ': 'Rz',
    }

    def __init__(
        self,
        size: int,
        device: Optional[Union[str, torch.device]] = None,
        threads: int = 8
    ):
        """
        Initialize a quantum circuit.

        Args:
            size: Number of qubits in the circuit.
            device: Computation device. Options:
                - 'gpu': Auto-detect CUDA or MPS
                - 'cpu': Use CPU with specified threads
                - torch.device: Use specific device
                - None: Default to CPU
            threads: Number of CPU threads (only used if device='cpu').

        Example:
            >>> qc = Circuit(4, device='gpu')  # Auto-detect GPU
            >>> qc = Circuit(4, device='cpu', threads=4)  # CPU with 4 threads
        """
        self.size = size
        self.device = self._configure_device(device, threads)
        self.gate_obj = Gate(device=self.device)

        # Gate lookup table mapping names to gate matrices/methods
        self._gate_registry = self._build_gate_registry()

        # Circuit state
        self.states: Optional[Tensor] = None
        self.measurements: Optional[Tensor] = None
        self.circuit: List[Dict[str, Any]] = []

    def _configure_device(
        self,
        device_type: Optional[Union[str, torch.device]],
        threads: int
    ) -> torch.device:
        """
        Configure the computation device.

        Args:
            device_type: Device specification ('gpu', 'cpu', or torch.device).
            threads: Number of CPU threads if using CPU.

        Returns:
            Configured torch.device.
        """
        if device_type is None:
            device_type = 'cpu'

        if isinstance(device_type, torch.device):
            return device_type

        device_str = str(device_type).lower()

        if device_str == 'gpu':
            if torch.cuda.is_available():
                print("Using CUDA GPU")
                return torch.device('cuda')
            elif torch.backends.mps.is_available():
                print("Using Apple Silicon MPS")
                return torch.device('mps')
            else:
                print("No GPU available, falling back to CPU")
                device_str = 'cpu'

        if device_str == 'cpu':
            torch.set_num_threads(threads)
            print(f"Using CPU with {threads} threads")
            return torch.device('cpu')

        # Assume it's a valid device string
        return torch.device(device_str)

    def _build_gate_registry(self) -> Dict[str, Any]:
        """
        Build a registry mapping gate names to gate matrices/methods.

        Returns:
            Dictionary mapping gate names to their implementations.
        """
        return {
            # Projection operators
            '0': self.gate_obj.zero,
            '1': self.gate_obj.one,

            # Single-qubit Clifford gates
            'H': self.gate_obj.H,
            'I': self.gate_obj.I,
            'X': self.gate_obj.X,
            'Y': self.gate_obj.Y,
            'Z': self.gate_obj.Z,
            'S': self.gate_obj.S,
            'Sdg': self.gate_obj.Sdg,
            'SX': self.gate_obj.SX,

            # T gates (non-Clifford)
            'T': self.gate_obj.T,
            'Tdg': self.gate_obj.Tdg,

            # Parametric gates (return callables)
            'P': self.gate_obj.P,
            'Ph': self.gate_obj.Ph,
            'Rx': self.gate_obj.Rx,
            'Ry': self.gate_obj.Ry,
            'Rz': self.gate_obj.Rz,
            'U': self.gate_obj.U,

            # Two-qubit gates (handled specially)
            'CNOT': 'CNOT',
            'CX': 'CNOT',
            'CZ': 'CZ',
            'SWAP': 'SWAP',

            # Measurement
            'MCM': 'MCM',
            'MEASURE': 'MCM',
        }

    def _normalize_gate_name(self, name: str) -> str:
        """Normalize gate name using aliases."""
        return self.GATE_ALIASES.get(name.upper(), name)

    # =========================================================================
    # Circuit Construction
    # =========================================================================

    def add(
        self,
        gate: str,
        target: int,
        control: Optional[int] = None,
        controls: Optional[List[int]] = None,
        angle: Optional[float] = None,
        angles: Optional[List[float]] = None
    ) -> 'Circuit':
        """
        Add a gate to the circuit.

        Args:
            gate: Gate name (e.g., 'H', 'X', 'CNOT', 'Rx').
            target: Target qubit index.
            control: Single control qubit (for two-qubit gates).
            controls: List of control qubits (for multi-controlled gates).
            angle: Single angle parameter (for Rx, Ry, Rz, P).
            angles: Multiple angle parameters (for U gate: [theta, phi, lambda]).

        Returns:
            self, for method chaining.

        Raises:
            ValueError: If gate name is unknown or parameters are invalid.

        Example:
            >>> qc = Circuit(3)
            >>> qc.add('H', 0).add('CNOT', target=1, control=0).add('Rx', 2, angle=math.pi/2)
        """
        gate_name = self._normalize_gate_name(gate)

        # Validate gate exists
        if gate_name not in self._gate_registry:
            raise ValueError(
                f"Unknown gate '{gate}'. Available gates: {list(self._gate_registry.keys())}"
            )

        # Validate target qubit
        if target < 0 or target >= self.size:
            raise ValueError(
                f"Target qubit {target} out of range for {self.size}-qubit circuit"
            )

        # Build gate metadata
        meta: Dict[str, Any] = {
            'name': gate_name,
            'target': target,
        }

        # Handle control qubits
        if control is not None:
            if controls is not None:
                raise ValueError("Specify either 'control' or 'controls', not both")
            controls = [control]

        if controls is not None:
            for c in controls:
                if c < 0 or c >= self.size:
                    raise ValueError(
                        f"Control qubit {c} out of range for {self.size}-qubit circuit"
                    )
                if c == target:
                    raise ValueError("Control and target qubits must be different")
            meta['controls'] = controls

        # Handle angle parameters
        if angle is not None:
            if angles is not None:
                raise ValueError("Specify either 'angle' or 'angles', not both")
            meta['angles'] = angle
        elif angles is not None:
            meta['angles'] = angles

        self.circuit.append(meta)
        return self

    def h(self, target: int) -> 'Circuit':
        """Add Hadamard gate. Shorthand for add('H', target)."""
        return self.add('H', target)

    def x(self, target: int) -> 'Circuit':
        """Add Pauli-X gate. Shorthand for add('X', target)."""
        return self.add('X', target)

    def y(self, target: int) -> 'Circuit':
        """Add Pauli-Y gate. Shorthand for add('Y', target)."""
        return self.add('Y', target)

    def z(self, target: int) -> 'Circuit':
        """Add Pauli-Z gate. Shorthand for add('Z', target)."""
        return self.add('Z', target)

    def s(self, target: int) -> 'Circuit':
        """Add S gate. Shorthand for add('S', target)."""
        return self.add('S', target)

    def t(self, target: int) -> 'Circuit':
        """Add T gate. Shorthand for add('T', target)."""
        return self.add('T', target)

    def rx(self, target: int, angle: float) -> 'Circuit':
        """Add Rx rotation gate."""
        return self.add('Rx', target, angle=angle)

    def ry(self, target: int, angle: float) -> 'Circuit':
        """Add Ry rotation gate."""
        return self.add('Ry', target, angle=angle)

    def rz(self, target: int, angle: float) -> 'Circuit':
        """Add Rz rotation gate."""
        return self.add('Rz', target, angle=angle)

    def p(self, target: int, angle: float) -> 'Circuit':
        """Add phase gate P(angle)."""
        return self.add('P', target, angle=angle)

    def u(self, target: int, theta: float, phi: float, lam: float) -> 'Circuit':
        """Add universal U gate with three Euler angles."""
        return self.add('U', target, angles=[theta, phi, lam])

    def cx(self, control: int, target: int) -> 'Circuit':
        """Add CNOT gate. Shorthand for add('CNOT', target, control=control)."""
        return self.add('CNOT', target, control=control)

    def cnot(self, control: int, target: int) -> 'Circuit':
        """Add CNOT gate. Alias for cx()."""
        return self.cx(control, target)

    def cz(self, control: int, target: int) -> 'Circuit':
        """Add CZ gate."""
        return self.add('CZ', target, control=control)

    def swap(self, qubit1: int, qubit2: int) -> 'Circuit':
        """Add SWAP gate."""
        return self.add('SWAP', qubit2, control=qubit1)

    def measure(self, target: int) -> 'Circuit':
        """Add mid-circuit measurement."""
        return self.add('MCM', target)

    def barrier(self) -> 'Circuit':
        """
        Add a barrier (no-op, for visual separation in circuit diagrams).

        Note: This is a no-op in simulation but useful for circuit organization.
        """
        self.circuit.append({'name': 'BARRIER', 'target': -1})
        return self

    # =========================================================================
    # Circuit Execution
    # =========================================================================

    def execute(self, shots: int = 1024, cache: bool = False) -> 'Circuit':
        """
        Execute the circuit and sample measurement outcomes.

        For circuits without mid-circuit measurements (MCM):
        - Runs the circuit once to obtain the final statevector
        - Samples from the probability distribution 'shots' times

        For circuits with mid-circuit measurements:
        - Runs the entire circuit 'shots' times (each run collapses differently)
        - Optionally caches the statevector up to the first MCM

        Args:
            shots: Number of measurement samples to take.
            cache: If True and MCM exists, cache statevector up to first MCM
                   to avoid redundant computation. Only beneficial when the
                   circuit has substantial work before the first MCM.

        Returns:
            self, for method chaining.

        Example:
            >>> qc = Circuit(2).add('H', 0).add('CNOT', target=1, control=0)
            >>> qc.execute(shots=1000)
            >>> print(qc.get_counts())
        """
        self.measurements = torch.zeros(
            2**self.size, dtype=torch.int64, device=self.device
        )

        mcm_idx = self._find_first_mcm()

        if mcm_idx is not None:
            # Circuit has mid-circuit measurement - must run multiple times
            if cache and mcm_idx > 0:
                # Cache statevector up to first MCM
                self._run_circuit(stop_idx=mcm_idx)
                cached_states = self.states.clone()

                for _ in range(shots):
                    self._run_circuit(states=cached_states, start_idx=mcm_idx)
                    outcome = self._sample_final_state(shots=1)
                    self.measurements += torch.bincount(
                        outcome, minlength=2**self.size
                    )
            else:
                # Run full circuit each time
                for _ in range(shots):
                    self._run_circuit()
                    outcome = self._sample_final_state(shots=1)
                    self.measurements += torch.bincount(
                        outcome, minlength=2**self.size
                    )
        else:
            # No MCM - run once and sample from final statevector
            self._run_circuit()
            outcomes = self._sample_final_state(shots=shots)
            self.measurements += torch.bincount(
                outcomes, minlength=2**self.size
            )

        return self

    def _find_first_mcm(self) -> Optional[int]:
        """Find index of first mid-circuit measurement, or None if none exist."""
        for i, gate in enumerate(self.circuit):
            if gate['name'] == 'MCM':
                return i
        return None

    def _run_circuit(
        self,
        states: Optional[Tensor] = None,
        start_idx: Optional[int] = None,
        stop_idx: Optional[int] = None
    ) -> None:
        """
        Execute gates in the circuit to compute the statevector.

        Args:
            states: Initial statevector (defaults to |0...0>).
            start_idx: Index to start execution from.
            stop_idx: Index to stop execution at (exclusive).

        Raises:
            ValueError: If indices are out of bounds.
        """
        start_idx = start_idx if start_idx is not None else 0
        stop_idx = stop_idx if stop_idx is not None else len(self.circuit)

        if start_idx > len(self.circuit) or stop_idx > len(self.circuit):
            raise ValueError("Start/stop indices out of bounds")
        if start_idx < 0 or stop_idx < 0:
            raise ValueError("Indices cannot be negative")

        # Initialize statevector
        if states is None:
            self.states = torch.zeros(
                2**self.size, dtype=torch.cfloat, device=self.device
            )
            self.states[0] = 1.0  # |0...0> state
        else:
            self.states = states.clone()

        # Apply each gate in sequence
        for gate_meta in self.circuit[start_idx:stop_idx]:
            self._apply_gate(gate_meta)

    def _apply_gate(self, gate_meta: Dict[str, Any]) -> None:
        """
        Apply a single gate operation to the statevector.

        Args:
            gate_meta: Dictionary containing gate name, target, and parameters.
        """
        name = gate_meta['name']
        target = gate_meta['target']

        # Skip barrier (no-op)
        if name == 'BARRIER':
            return

        # Handle two-qubit gates
        if name == 'CNOT':
            controls = gate_meta.get('controls', [])
            if not controls:
                raise ValueError("CNOT requires a control qubit")
            self.gate_obj.CNOT(self.states, controls[0], target)
            return

        if name == 'CZ':
            controls = gate_meta.get('controls', [])
            if not controls:
                raise ValueError("CZ requires a control qubit")
            self.gate_obj.CZ(self.states, controls[0], target)
            return

        if name == 'SWAP':
            controls = gate_meta.get('controls', [])
            if not controls:
                raise ValueError("SWAP requires two qubits")
            self.gate_obj.SWAP(self.states, controls[0], target)
            return

        # Handle mid-circuit measurement
        if name == 'MCM':
            self.gate_obj.measure(self.states, target)
            return

        # Get gate matrix
        gate_entry = self._gate_registry[name]

        if callable(gate_entry):
            # Parametric gate - need to call with angles
            angles = gate_meta.get('angles')
            if angles is None:
                raise ValueError(f"Gate '{name}' requires angle parameter(s)")

            if isinstance(angles, (list, tuple)):
                gate_matrix = gate_entry(*angles)
            else:
                gate_matrix = gate_entry(angles)
        else:
            # Fixed gate matrix
            gate_matrix = gate_entry

        # Apply the gate
        controls = gate_meta.get('controls')
        self.gate_obj.apply(gate_matrix, self.states, target, controls)

    def _sample_final_state(self, shots: int) -> Tensor:
        """
        Sample measurement outcomes from the statevector.

        Args:
            shots: Number of samples to take.

        Returns:
            Tensor of measurement outcomes (indices into the statevector).
        """
        probs = self.states.abs().pow(2)
        cdf = probs.cumsum(dim=0)
        random_vals = torch.rand(shots, device=self.device)
        return torch.searchsorted(cdf, random_vals)

    # =========================================================================
    # Results and Analysis
    # =========================================================================

    def get_statevector(self) -> Optional[Tensor]:
        """
        Get the final statevector after circuit execution.

        Returns:
            The statevector tensor, or None if circuit hasn't been executed.
        """
        return self.states

    def get_probabilities(self) -> Optional[Tensor]:
        """
        Get the probability distribution from the statevector.

        Returns:
            Tensor of probabilities for each basis state.
        """
        if self.states is None:
            return None
        return self.states.abs().pow(2)

    def get_counts(self) -> Optional[Dict[str, int]]:
        """
        Get measurement counts as a dictionary.

        Returns:
            Dictionary mapping bitstrings to counts, or None if not executed.

        Example:
            >>> qc.execute(shots=1000)
            >>> counts = qc.get_counts()
            >>> print(counts)  # {'00': 498, '11': 502}
        """
        if self.measurements is None:
            return None

        counts = {}
        for i, count in enumerate(self.measurements.cpu().numpy()):
            if count > 0:
                # Convert index to bitstring (little-endian)
                bitstring = format(i, f'0{self.size}b')[::-1]
                counts[bitstring] = int(count)

        return counts

    def get_counts_big_endian(self) -> Optional[Dict[str, int]]:
        """
        Get measurement counts with big-endian bitstrings.

        In big-endian format, qubit 0 is the rightmost bit.

        Returns:
            Dictionary mapping bitstrings to counts.
        """
        if self.measurements is None:
            return None

        counts = {}
        for i, count in enumerate(self.measurements.cpu().numpy()):
            if count > 0:
                bitstring = format(i, f'0{self.size}b')
                counts[bitstring] = int(count)

        return counts

    # =========================================================================
    # Circuit Information
    # =========================================================================

    def depth(self) -> int:
        """
        Calculate the circuit depth (number of gate layers).

        Returns:
            Circuit depth as an integer.
        """
        if not self.circuit:
            return 0

        # Track when each qubit becomes free
        qubit_depth = [0] * self.size

        for gate_meta in self.circuit:
            if gate_meta['name'] == 'BARRIER':
                continue

            target = gate_meta['target']
            controls = gate_meta.get('controls', [])

            # Find the maximum depth among all involved qubits
            involved_qubits = [target] + controls
            max_depth = max(qubit_depth[q] for q in involved_qubits)

            # Update all involved qubits to new depth
            new_depth = max_depth + 1
            for q in involved_qubits:
                qubit_depth[q] = new_depth

        return max(qubit_depth)

    def gate_count(self) -> Dict[str, int]:
        """
        Count occurrences of each gate type.

        Returns:
            Dictionary mapping gate names to counts.
        """
        counts: Dict[str, int] = {}
        for gate_meta in self.circuit:
            name = gate_meta['name']
            if name == 'BARRIER':
                continue
            counts[name] = counts.get(name, 0) + 1
        return counts

    def __len__(self) -> int:
        """Return the number of gates in the circuit."""
        return len([g for g in self.circuit if g['name'] != 'BARRIER'])

    def __repr__(self) -> str:
        """String representation of the circuit."""
        return f"Circuit(qubits={self.size}, gates={len(self)}, depth={self.depth()})"

    def print_circuit(self) -> None:
        """Print a text representation of the circuit gates."""
        print(f"Circuit with {self.size} qubits:")
        print("-" * 40)
        for i, gate in enumerate(self.circuit):
            if gate['name'] == 'BARRIER':
                print(f"  [{i}] ──── BARRIER ────")
                continue

            line = f"  [{i}] {gate['name']}"
            line += f" @ q{gate['target']}"

            if 'controls' in gate:
                ctrl_str = ', '.join(f"q{c}" for c in gate['controls'])
                line += f" (ctrl: {ctrl_str})"

            if 'angles' in gate:
                angles = gate['angles']
                if isinstance(angles, (list, tuple)):
                    angle_str = ', '.join(f"{a:.4f}" for a in angles)
                else:
                    angle_str = f"{angles:.4f}"
                line += f" (angles: {angle_str})"

            print(line)
        print("-" * 40)

    def reset(self) -> 'Circuit':
        """
        Clear the circuit and reset state.

        Returns:
            self, for method chaining.
        """
        self.circuit = []
        self.states = None
        self.measurements = None
        return self

    def copy(self) -> 'Circuit':
        """
        Create a copy of this circuit.

        Returns:
            A new Circuit with the same gates.
        """
        new_circuit = Circuit(self.size, device=self.device)
        new_circuit.circuit = [gate.copy() for gate in self.circuit]
        return new_circuit

    def change_device(self, new_device: Union[str, torch.device]) -> 'Circuit':
        """
        Change the computation device.

        Args:
            new_device: New device specification.

        Returns:
            self, for method chaining.
        """
        if isinstance(new_device, str):
            new_device = torch.device(new_device)

        self.device = new_device
        self.gate_obj = Gate(device=new_device)
        self._gate_registry = self._build_gate_registry()

        # Move existing state tensors if they exist
        if self.states is not None:
            self.states = self.states.to(new_device)
        if self.measurements is not None:
            self.measurements = self.measurements.to(new_device)

        return self
