"""
Example usage of the mps_accel_qc quantum circuit simulator.

This module demonstrates:
1. Creating quantum circuits
2. Building common quantum states (Bell state, GHZ state)
3. Using rotation gates
4. Mid-circuit measurements
5. Analyzing results
6. Circuit visualization (ASCII and matplotlib)

Run examples with:
    python -m mps_accel_qc.examples

Or from Python:
    from mps_accel_qc import run_examples
    run_examples()
"""

import math

from .circuit import Circuit
from .visualization import CircuitVisualizer, print_circuit
from .utils import (
    get_available_devices,
    get_best_device,
    print_statevector,
    print_memory_stats,
    counts_to_probabilities,
)


def run_examples():
    """
    Run all example circuits demonstrating the mps_accel_qc library.

    Examples include:
    - Bell state creation
    - GHZ state (3-qubit entanglement)
    - Rotation gates
    - Quantum teleportation setup
    - Uniform superposition
    - Method chaining API
    - Circuit visualization
    """
    # Check available devices
    print("=" * 60)
    print("Available Devices:")
    print("=" * 60)
    devices = get_available_devices()
    for device, available in devices.items():
        status = "available" if available else "not available"
        print(f"  {device}: {status}")
    print(f"  Best device: {get_best_device()}")
    print()

    # -----------------------------------------------------
    # Example 1: Bell State
    # -----------------------------------------------------
    print("=" * 60)
    print("Example 1: Bell State")
    print("=" * 60)

    # Create a 2-qubit circuit
    bell = Circuit(2, device='cpu')

    # Build Bell state: |00> + |11> / sqrt(2)
    bell.h(0)          # Hadamard on qubit 0
    bell.cnot(0, 1)    # CNOT with control=0, target=1

    # Show circuit using ASCII visualization
    print("\nCircuit Diagram (ASCII):")
    print("-" * 40)
    print_circuit(bell)
    print("-" * 40)

    # Execute with 1000 shots
    bell.execute(shots=1000)

    # Display results
    print("\nMeasurement counts:")
    counts = bell.get_counts()
    for state, count in sorted(counts.items()):
        print(f"  |{state}>: {count}")

    print("\nProbabilities:")
    probs = counts_to_probabilities(counts)
    for state, prob in sorted(probs.items()):
        print(f"  |{state}>: {prob:.3f}")

    print("\nStatevector (before measurement collapse):")
    bell._run_circuit()  # Re-run to get clean statevector
    print_statevector(bell.states)

    # -----------------------------------------------------
    # Example 2: GHZ State (3 qubits)
    # -----------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 2: GHZ State (3 qubits)")
    print("=" * 60)

    # GHZ state: |000> + |111> / sqrt(2)
    ghz = Circuit(3, device='cpu')
    ghz.h(0)
    ghz.cnot(0, 1)
    ghz.cnot(1, 2)

    # Show circuit
    print("\nCircuit Diagram (ASCII):")
    print("-" * 40)
    print_circuit(ghz)
    print("-" * 40)

    ghz.execute(shots=1000)

    print("\nMeasurement counts:")
    for state, count in sorted(ghz.get_counts().items()):
        print(f"  |{state}>: {count}")

    # -----------------------------------------------------
    # Example 3: Rotation Gates
    # -----------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 3: Rotation Gates")
    print("=" * 60)

    rot = Circuit(1, device='cpu')

    # Rotate around Y-axis by pi/4 (creates superposition)
    rot.ry(0, math.pi / 4)

    # Show circuit
    print("\nCircuit Diagram (ASCII):")
    print("-" * 40)
    print_circuit(rot, show_angles=True)
    print("-" * 40)

    rot.execute(shots=1000)

    print("\nRy(pi/4) on |0>:")
    print("Expected: ~85% |0>, ~15% |1>")
    print("Measured:")
    for state, count in sorted(rot.get_counts().items()):
        print(f"  |{state}>: {count} ({count/10:.1f}%)")

    # -----------------------------------------------------
    # Example 4: Quantum Teleportation Circuit
    # -----------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 4: Quantum Teleportation Setup")
    print("=" * 60)

    # 3 qubits: q0 = state to teleport, q1 & q2 = entangled pair
    teleport = Circuit(3, device='cpu')

    # Prepare state to teleport (arbitrary rotation)
    teleport.rx(0, math.pi / 3)

    # Create entangled pair between q1 and q2
    teleport.h(1)
    teleport.cnot(1, 2)

    # Bell measurement on q0 and q1
    teleport.cnot(0, 1)
    teleport.h(0)

    # Show circuit
    print("\nCircuit Diagram (ASCII):")
    print("-" * 40)
    print_circuit(teleport, show_angles=True)
    print("-" * 40)

    print(f"\nCircuit depth: {teleport.depth()}")
    print(f"Gate counts: {teleport.gate_count()}")

    teleport.execute(shots=1000)
    print("\nMeasurement distribution:")
    for state, count in sorted(teleport.get_counts().items()):
        print(f"  |{state}>: {count}")

    # -----------------------------------------------------
    # Example 5: Superposition of all states
    # -----------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 5: Uniform Superposition (4 qubits)")
    print("=" * 60)

    uniform = Circuit(4, device='cpu')

    # Apply Hadamard to all qubits
    for i in range(4):
        uniform.h(i)

    # Show circuit
    print("\nCircuit Diagram (ASCII):")
    print("-" * 40)
    print_circuit(uniform)
    print("-" * 40)

    uniform.execute(shots=4000)

    print("\nAll 16 states should have ~250 counts each:")
    counts = uniform.get_counts()
    for state, count in sorted(counts.items()):
        bar = "#" * (count // 25)
        print(f"  |{state}>: {count:4d} {bar}")

    # -----------------------------------------------------
    # Example 6: Method Chaining
    # -----------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 6: Fluent API (Method Chaining)")
    print("=" * 60)

    # Build and execute in one chain
    result = (
        Circuit(2, device='cpu')
        .h(0)
        .cnot(0, 1)
        .z(1)
        .execute(shots=500)
    )

    print("\nCircuit Diagram (ASCII):")
    print("-" * 40)
    print_circuit(result)
    print("-" * 40)

    print("\nResults:", result.get_counts())

    # -----------------------------------------------------
    # Example 7: Full Circuit Summary
    # -----------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 7: Complete Circuit Summary")
    print("=" * 60)

    # Build a more complex circuit
    complex_circuit = Circuit(3, device='cpu')
    complex_circuit.h(0)
    complex_circuit.h(1)
    complex_circuit.cnot(0, 2)
    complex_circuit.cnot(1, 2)
    complex_circuit.t(2)
    complex_circuit.s(0)
    complex_circuit.execute(shots=1000)

    # Use CircuitVisualizer for complete summary
    vis = CircuitVisualizer(complex_circuit)
    vis.summary()

    # -----------------------------------------------------
    # Example 8: Saving Circuit Diagrams (optional)
    # -----------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 8: Matplotlib Visualization")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend

        # Create a circuit for visualization
        viz_circuit = Circuit(3, device='cpu')
        viz_circuit.h(0)
        viz_circuit.cnot(0, 1)
        viz_circuit.cnot(1, 2)
        viz_circuit.rz(2, math.pi/4)
        viz_circuit.execute(shots=1000)

        vis = CircuitVisualizer(viz_circuit)

        # Save circuit diagram
        vis.save_circuit("circuit_diagram.png", title="GHZ Circuit with Rz")
        print("Circuit diagram saved to: circuit_diagram.png")

        # Save probability plot
        vis.save_probabilities("probabilities.png", title="State Probabilities")
        print("Probability plot saved to: probabilities.png")

        # Save measurement counts
        vis.save_counts("counts.png", title="Measurement Results")
        print("Counts plot saved to: counts.png")

    except ImportError:
        print("matplotlib not installed - skipping image generation")
        print("Install with: pip install matplotlib")

    # -----------------------------------------------------
    # Memory stats
    # -----------------------------------------------------
    print("\n" + "=" * 60)
    print("Memory Statistics")
    print("=" * 60)
    print_memory_stats()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


# Allow running as a module: python -m mps_accel_qc.examples
if __name__ == "__main__":
    run_examples()
