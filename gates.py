"""
Quantum Gate Definitions for Statevector Simulation.

This module provides the Gate class which contains all standard quantum gates
and gate application routines for statevector simulation. Supports CPU, CUDA,
and Apple Silicon (MPS) backends via PyTorch.

Author: Nathaniel Sun
"""

import math
from typing import Optional, List

import torch
from torch import Tensor


class Gate:
    """
    A collection of quantum gates and gate application methods.

    This class provides standard single-qubit gates, rotation gates,
    controlled gates, and measurement operations for quantum circuit
    simulation using statevector representation.

    Attributes:
        device: The torch device (cpu, cuda, mps) for tensor operations.

    Example:
        >>> gate = Gate(device=torch.device('cpu'))
        >>> state = torch.tensor([1.0+0j, 0.0+0j])  # |0> state
        >>> gate.apply(gate.H, state, target=0)     # Apply Hadamard
        >>> print(state)  # Now in |+> state
    """

    def __init__(self, device: torch.device):
        """
        Initialize the Gate object with standard quantum gates.

        Args:
            device: PyTorch device for tensor allocation (cpu, cuda, mps).
        """
        self.device = device
        self._initialize_gates()

    def _initialize_gates(self) -> None:
        """Initialize all standard quantum gate matrices."""

        # Projection operators for measurement
        self.zero = torch.tensor(
            [[1., 0.], [0., 0.]],
            device=self.device, dtype=torch.cfloat
        )
        self.one = torch.tensor(
            [[0., 0.], [0., 1.]],
            device=self.device, dtype=torch.cfloat
        )

        # Hadamard gate: creates superposition
        # H|0> = |+> = (|0> + |1>)/sqrt(2)
        # H|1> = |-> = (|0> - |1>)/sqrt(2)
        self.H = torch.tensor(
            [[1., 1.], [1., -1.]],
            device=self.device, dtype=torch.cfloat
        ) / math.sqrt(2)

        # Identity gate
        self.I = torch.tensor(
            [[1.+0.j, 0.+0.j], [0.+0.j, 1.+0.j]],
            device=self.device, dtype=torch.cfloat
        )

        # Pauli-X gate (NOT gate, bit flip)
        # X|0> = |1>, X|1> = |0>
        self.X = torch.tensor(
            [[0., 1.], [1., 0.]],
            device=self.device, dtype=torch.cfloat
        )

        # Pauli-Y gate
        # Y|0> = i|1>, Y|1> = -i|0>
        self.Y = torch.tensor(
            [[0., -1.j], [1.j, 0.]],
            device=self.device, dtype=torch.cfloat
        )

        # Pauli-Z gate (phase flip)
        # Z|0> = |0>, Z|1> = -|1>
        self.Z = torch.tensor(
            [[1., 0.], [0., -1.]],
            device=self.device, dtype=torch.cfloat
        )

        # S gate (sqrt(Z), phase gate with pi/2 rotation)
        # S|0> = |0>, S|1> = i|1>
        self.S = torch.tensor(
            [[1., 0.], [0., 1.j]],
            device=self.device, dtype=torch.cfloat
        )

        # S-dagger gate (inverse of S)
        self.Sdg = torch.tensor(
            [[1., 0.], [0., -1.j]],
            device=self.device, dtype=torch.cfloat
        )

        # T gate (sqrt(S), fourth root of Z, pi/8 gate)
        # T|0> = |0>, T|1> = e^(i*pi/4)|1>
        self.T = self.P(math.pi / 4)

        # T-dagger gate (inverse of T)
        self.Tdg = self.P(-math.pi / 4)

        # sqrt(X) gate (square root of X)
        self.SX = torch.tensor(
            [[1.+1.j, 1.-1.j], [1.-1.j, 1.+1.j]],
            device=self.device, dtype=torch.cfloat
        ) / 2

    # =========================================================================
    # Single-Qubit Gate Application
    # =========================================================================

    def apply(
        self,
        gate: Tensor,
        state: Tensor,
        target: int,
        controls: Optional[List[int]] = None
    ) -> None:
        """
        Apply a single-qubit gate to the statevector using strided indexing.

        This method efficiently applies a 2x2 gate matrix to a specific qubit
        in the statevector without constructing large tensor products. It
        supports optional control qubits for implementing controlled gates.

        The algorithm:
        1. Identifies all index pairs (i, j) where i and j differ only in
           the target qubit bit.
        2. If controls are specified, filters to only indices where all
           control qubits are |1>.
        3. Applies the 2x2 transformation to each pair of amplitudes.

        Args:
            gate: A 2x2 complex tensor representing the gate matrix.
            state: The statevector to modify (modified in-place).
            target: Index of the target qubit (0-indexed from LSB).
            controls: Optional list of control qubit indices.

        Raises:
            ValueError: If target index is negative, out of range, or if
                the statevector length is not a power of two.

        Note:
            The qubit indexing follows little-endian convention where
            qubit 0 is the least significant bit.
        """
        N = state.numel()
        num_qubits = int(math.log2(N))

        # Input validation
        if target < 0:
            raise ValueError("Target qubit index cannot be negative")
        if target >= num_qubits:
            raise ValueError(
                f"Target qubit {target} out of range for {num_qubits}-qubit system"
            )
        if N & (N - 1):
            raise ValueError("Statevector length must be a power of two")

        # Bit mask for selecting the target qubit
        target_bit = 1 << target

        # Generate all base indices where target bit = 0
        all_indices = torch.arange(N, device=state.device)
        idx0 = all_indices[(all_indices & target_bit) == 0]

        # Apply control mask if controls are specified
        if controls:
            control_mask = sum(1 << c for c in controls)
            # Keep only indices where all control bits are 1
            idx0 = idx0[(idx0 & control_mask) == control_mask]

        # Corresponding indices where target bit = 1
        idx1 = idx0 | target_bit

        # Extract amplitude pairs
        v0 = state[idx0]
        v1 = state[idx1]

        # Apply 2x2 gate transformation
        # |psi'> = G|psi> where G is the gate matrix
        out0 = gate[0, 0] * v0 + gate[0, 1] * v1
        out1 = gate[1, 0] * v0 + gate[1, 1] * v1

        # Write back results
        state[idx0] = out0
        state[idx1] = out1

    # =========================================================================
    # Rotation Gates
    # =========================================================================

    def Ph(self, phase: float) -> Tensor:
        """
        Create a global phase gate.

        Applies a global phase e^(i*phi) to the qubit state. This doesn't
        change measurement probabilities but is important for interference
        effects in multi-qubit systems.

        Args:
            phase: The phase angle in radians.

        Returns:
            A 2x2 tensor representing the phase gate.
        """
        phase = phase % (2 * math.pi)
        phase_factor = torch.exp(torch.tensor(1j * phase, device=self.device))
        return phase_factor * self.I

    def P(self, phi: float) -> Tensor:
        """
        Create a phase shift gate (P or R_phi gate).

        Applies a phase shift to the |1> component:
        P(phi)|0> = |0>
        P(phi)|1> = e^(i*phi)|1>

        Args:
            phi: The phase angle in radians.

        Returns:
            A 2x2 tensor representing the phase gate.

        Note:
            Special cases: P(pi/2) = S, P(pi/4) = T, P(pi) = Z
        """
        phi_exp = torch.exp(torch.tensor(1j * phi, device=self.device))
        return torch.tensor(
            [[1, 0], [0, phi_exp]],
            dtype=torch.cfloat, device=self.device
        )

    def Rx(self, angle: float) -> Tensor:
        """
        Create an X-axis rotation gate.

        Rotates the qubit state around the X-axis of the Bloch sphere:
        Rx(theta) = cos(theta/2)*I - i*sin(theta/2)*X

        Args:
            angle: The rotation angle in radians.

        Returns:
            A 2x2 tensor representing the Rx gate.
        """
        theta = torch.tensor(angle / 2, device=self.device)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        return torch.tensor(
            [[cos_t, -1j * sin_t], [-1j * sin_t, cos_t]],
            dtype=torch.cfloat, device=self.device
        )

    def Ry(self, angle: float) -> Tensor:
        """
        Create a Y-axis rotation gate.

        Rotates the qubit state around the Y-axis of the Bloch sphere:
        Ry(theta) = cos(theta/2)*I - i*sin(theta/2)*Y

        Args:
            angle: The rotation angle in radians.

        Returns:
            A 2x2 tensor representing the Ry gate.
        """
        theta = torch.tensor(angle / 2, device=self.device)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        return torch.tensor(
            [[cos_t, -sin_t], [sin_t, cos_t]],
            dtype=torch.cfloat, device=self.device
        )

    def Rz(self, angle: float) -> Tensor:
        """
        Create a Z-axis rotation gate.

        Rotates the qubit state around the Z-axis of the Bloch sphere:
        Rz(theta) = cos(theta/2)*I - i*sin(theta/2)*Z = diag(e^(-i*theta/2), e^(i*theta/2))

        Args:
            angle: The rotation angle in radians.

        Returns:
            A 2x2 tensor representing the Rz gate.
        """
        theta = torch.tensor(angle / 2, device=self.device)
        return torch.tensor(
            [[torch.exp(-1j * theta), 0], [0, torch.exp(1j * theta)]],
            dtype=torch.cfloat, device=self.device
        )

    def U(self, theta: float, phi: float, lam: float) -> Tensor:
        """
        Create a general single-qubit unitary gate (U3 gate).

        This is the most general single-qubit gate, parameterized by three
        Euler angles. Any single-qubit unitary can be expressed as U(theta, phi, lambda).

        U(theta, phi, lambda) = [[cos(theta/2), -e^(i*lambda)*sin(theta/2)],
                                 [e^(i*phi)*sin(theta/2), e^(i*(phi+lambda))*cos(theta/2)]]

        Args:
            theta: Rotation angle around Y-axis (in radians).
            phi: Phase angle for |1> -> |0> transition.
            lam: Phase angle for |0> -> |1> transition.

        Returns:
            A 2x2 tensor representing the U gate.

        Note:
            Special cases:
            - U(theta, -pi/2, pi/2) = Rx(theta)
            - U(theta, 0, 0) = Ry(theta)
            - U(0, 0, lambda) = Rz(lambda) (up to global phase)
        """
        theta_t = torch.tensor(theta / 2, device=self.device)
        phi_t = torch.tensor(phi, device=self.device)
        lam_t = torch.tensor(lam, device=self.device)

        cos_t = torch.cos(theta_t)
        sin_t = torch.sin(theta_t)

        return torch.tensor([
            [cos_t, -torch.exp(1j * lam_t) * sin_t],
            [torch.exp(1j * phi_t) * sin_t, torch.exp(1j * (lam_t + phi_t)) * cos_t]
        ], dtype=torch.cfloat, device=self.device)

    # =========================================================================
    # Two-Qubit Gates
    # =========================================================================

    def CNOT(self, state: Tensor, control: int, target: int) -> None:
        """
        Apply a CNOT (controlled-X) gate.

        Flips the target qubit if and only if the control qubit is |1>:
        CNOT|00> = |00>
        CNOT|01> = |01>
        CNOT|10> = |11>
        CNOT|11> = |10>

        Uses efficient bitwise operations rather than matrix multiplication.

        Args:
            state: The statevector to modify (modified in-place).
            control: Index of the control qubit.
            target: Index of the target qubit.

        Raises:
            ValueError: If qubit indices are negative or out of range.
        """
        N = state.numel()
        num_qubits = int(math.log2(N))

        if control < 0 or target < 0:
            raise ValueError("Qubit indices cannot be negative")
        if control >= num_qubits or target >= num_qubits:
            raise ValueError(
                f"Qubit indices out of range for {num_qubits}-qubit system"
            )
        if control == target:
            raise ValueError("Control and target qubits must be different")

        control_bit = 1 << control
        target_bit = 1 << target

        # Find indices where control qubit is |1>
        indices = torch.arange(N, device=state.device)
        controlled_indices = indices[(indices & control_bit) != 0]

        # For each controlled index, compute the index with target flipped
        flipped_indices = controlled_indices ^ target_bit

        # Swap amplitudes (implementing X on target conditioned on control)
        temp = state[controlled_indices].clone()
        state[controlled_indices] = state[flipped_indices]
        state[flipped_indices] = temp

    def CZ(self, state: Tensor, control: int, target: int) -> None:
        """
        Apply a CZ (controlled-Z) gate.

        Applies a phase flip to |11>:
        CZ|00> = |00>
        CZ|01> = |01>
        CZ|10> = |10>
        CZ|11> = -|11>

        Args:
            state: The statevector to modify (modified in-place).
            control: Index of the control qubit.
            target: Index of the target qubit.
        """
        N = state.numel()
        num_qubits = int(math.log2(N))

        if control < 0 or target < 0:
            raise ValueError("Qubit indices cannot be negative")
        if control >= num_qubits or target >= num_qubits:
            raise ValueError(
                f"Qubit indices out of range for {num_qubits}-qubit system"
            )

        # Mask for both qubits being |1>
        both_one_mask = (1 << control) | (1 << target)

        indices = torch.arange(N, device=state.device)
        affected = indices[(indices & both_one_mask) == both_one_mask]

        # Apply phase flip
        state[affected] *= -1

    def SWAP(self, state: Tensor, qubit1: int, qubit2: int) -> None:
        """
        Apply a SWAP gate between two qubits.

        Exchanges the states of two qubits:
        SWAP|01> = |10>
        SWAP|10> = |01>

        Args:
            state: The statevector to modify (modified in-place).
            qubit1: Index of the first qubit.
            qubit2: Index of the second qubit.
        """
        N = state.numel()
        num_qubits = int(math.log2(N))

        if qubit1 < 0 or qubit2 < 0:
            raise ValueError("Qubit indices cannot be negative")
        if qubit1 >= num_qubits or qubit2 >= num_qubits:
            raise ValueError(
                f"Qubit indices out of range for {num_qubits}-qubit system"
            )
        if qubit1 == qubit2:
            return  # No-op

        bit1 = 1 << qubit1
        bit2 = 1 << qubit2

        # Find indices where exactly one of the two qubits is |1>
        indices = torch.arange(N, device=state.device)

        # Indices where qubit1=0, qubit2=1
        mask_01 = ((indices & bit1) == 0) & ((indices & bit2) != 0)
        idx_01 = indices[mask_01]

        # Corresponding indices where qubit1=1, qubit2=0
        idx_10 = (idx_01 | bit1) & ~bit2

        # Swap amplitudes
        temp = state[idx_01].clone()
        state[idx_01] = state[idx_10]
        state[idx_10] = temp

    # =========================================================================
    # Measurement
    # =========================================================================

    def measure(self, state: Tensor, target: int) -> int:
        """
        Perform a mid-circuit measurement on a single qubit.

        Collapses the statevector according to the Born rule:
        1. Computes probability of measuring |1> on target qubit.
        2. Samples measurement outcome based on this probability.
        3. Collapses statevector to the post-measurement state.
        4. Renormalizes the surviving amplitudes.

        Args:
            state: The statevector to measure (modified in-place).
            target: Index of the qubit to measure.

        Returns:
            The measurement result (0 or 1).

        Raises:
            ValueError: If target qubit index is out of range.

        Note:
            This operation is non-unitary and irreversible. The statevector
            is collapsed to the post-measurement state.
        """
        N = state.numel()
        num_qubits = int(math.log2(N))

        if target < 0 or target >= num_qubits:
            raise ValueError(
                f"Target qubit {target} out of range for {num_qubits}-qubit system"
            )

        # Bit mask for selecting |1> subspace of target qubit
        target_bit = 1 << target
        indices = torch.arange(N, device=state.device)
        mask_one = (indices & target_bit) != 0

        # Compute probability of measuring |1>
        prob_one = state[mask_one].abs().pow(2).sum().item()

        # Sample measurement outcome
        outcome = 1 if torch.rand(()).item() < prob_one else 0

        if outcome == 1:
            # Collapse to |1> subspace
            state[~mask_one] = 0.0
            if prob_one > 0:
                state /= math.sqrt(prob_one)
        else:
            # Collapse to |0> subspace
            state[mask_one] = 0.0
            prob_zero = 1.0 - prob_one
            if prob_zero > 0:
                state /= math.sqrt(prob_zero)

        return outcome
