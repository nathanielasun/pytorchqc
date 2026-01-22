"""
Circuit Visualization Module.

This module provides pretty-printing and image generation for quantum circuits.
Supports both terminal (ASCII) output and matplotlib-based image generation.

Features:
- ASCII circuit diagrams for terminal display
- Matplotlib circuit diagram images
- Probability distribution visualization
- Statevector amplitude plots

Author: Nathaniel Sun
"""

import math
from typing import Optional, Dict, List, Tuple, Any, TYPE_CHECKING
from pathlib import Path

import torch

if TYPE_CHECKING:
    from .circuit import Circuit


# =============================================================================
# ASCII Circuit Drawing (Terminal)
# =============================================================================

class ASCIICircuitDrawer:
    """
    Draws quantum circuits as ASCII art for terminal display.

    Example output:
        q0: ──H──●──────X──M──
                │      │
        q1: ────X──●───┼─────
                   │   │
        q2: ───────X───●─────
    """

    # Gate display symbols
    GATE_SYMBOLS = {
        'H': 'H',
        'X': 'X',
        'Y': 'Y',
        'Z': 'Z',
        'S': 'S',
        'Sdg': 'S†',
        'T': 'T',
        'Tdg': 'T†',
        'SX': '√X',
        'I': 'I',
        'CNOT': 'X',
        'CX': 'X',
        'CZ': 'Z',
        'SWAP': '×',
        'MCM': 'M',
        'MEASURE': 'M',
        'Rx': 'Rx',
        'Ry': 'Ry',
        'Rz': 'Rz',
        'P': 'P',
        'Ph': 'Ph',
        'U': 'U',
        '0': '|0⟩',
        '1': '|1⟩',
    }

    def __init__(self, circuit: 'Circuit'):
        """
        Initialize the ASCII drawer.

        Args:
            circuit: The Circuit object to draw.
        """
        self.circuit = circuit
        self.num_qubits = circuit.size

    def draw(self, show_angles: bool = True) -> str:
        """
        Generate ASCII representation of the circuit.

        Args:
            show_angles: Whether to display angle parameters.

        Returns:
            String containing the ASCII circuit diagram.
        """
        if not self.circuit.circuit:
            return self._empty_circuit()

        # Build layers (gates that can be drawn in parallel)
        layers = self._build_layers()

        # Initialize wire lines for each qubit
        lines = {q: [] for q in range(self.num_qubits)}
        connector_lines = {q: [] for q in range(self.num_qubits)}

        # Draw each layer
        for layer in layers:
            layer_width = self._get_layer_width(layer, show_angles)
            self._draw_layer(layer, lines, connector_lines, layer_width, show_angles)

        # Assemble final output
        return self._assemble_output(lines, connector_lines)

    def _empty_circuit(self) -> str:
        """Generate display for empty circuit."""
        lines = []
        for q in range(self.num_qubits):
            lines.append(f"q{q}: {'─' * 20}")
        return '\n'.join(lines)

    def _build_layers(self) -> List[List[Dict[str, Any]]]:
        """
        Organize gates into layers for parallel display.

        Returns:
            List of layers, each containing gates that don't overlap.
        """
        layers: List[List[Dict[str, Any]]] = []
        qubit_depth = [0] * self.num_qubits

        for gate in self.circuit.circuit:
            if gate['name'] == 'BARRIER':
                # Barrier forces new layer
                max_depth = max(qubit_depth)
                qubit_depth = [max_depth] * self.num_qubits
                continue

            target = gate['target']
            controls = gate.get('controls', [])
            involved = [target] + controls

            # Find the layer this gate should go in
            layer_idx = max(qubit_depth[q] for q in involved)

            # Ensure layer exists
            while len(layers) <= layer_idx:
                layers.append([])

            layers[layer_idx].append(gate)

            # Update depths
            for q in involved:
                qubit_depth[q] = layer_idx + 1

        return layers

    def _get_layer_width(self, layer: List[Dict], show_angles: bool) -> int:
        """Calculate the display width needed for a layer."""
        max_width = 3  # Minimum width

        for gate in layer:
            name = gate['name']
            symbol = self.GATE_SYMBOLS.get(name, name[:3])
            width = len(symbol) + 2

            if show_angles and 'angles' in gate:
                angles = gate['angles']
                if isinstance(angles, (list, tuple)):
                    angle_str = ','.join(f'{a:.2f}' for a in angles)
                else:
                    angle_str = f'{angles:.2f}'
                width = max(width, len(symbol) + len(angle_str) + 4)

            max_width = max(max_width, width)

        return max_width

    def _draw_layer(
        self,
        layer: List[Dict],
        lines: Dict[int, List[str]],
        connector_lines: Dict[int, List[str]],
        width: int,
        show_angles: bool
    ) -> None:
        """Draw a single layer of gates."""
        # Track which qubits have gates in this layer
        qubit_content = {q: None for q in range(self.num_qubits)}
        connections = []  # (min_qubit, max_qubit, control_qubits, target)

        for gate in layer:
            target = gate['target']
            controls = gate.get('controls', [])
            name = gate['name']

            # Get display symbol
            symbol = self.GATE_SYMBOLS.get(name, name[:3])

            # Add angle if present
            if show_angles and 'angles' in gate:
                angles = gate['angles']
                if isinstance(angles, (list, tuple)):
                    angle_str = ','.join(f'{a:.1f}' for a in angles)
                else:
                    angle_str = f'{angles:.2f}'
                symbol = f'{symbol}({angle_str})'

            # Store gate content
            qubit_content[target] = ('gate', symbol)

            # Handle controls
            for c in controls:
                qubit_content[c] = ('control', '●')

            # Track connections needed
            if controls:
                all_qubits = [target] + controls
                connections.append((min(all_qubits), max(all_qubits), controls, target))

            # Handle SWAP specially
            if name == 'SWAP' and controls:
                qubit_content[controls[0]] = ('gate', '×')
                qubit_content[target] = ('gate', '×')

        # Draw the layer
        for q in range(self.num_qubits):
            content = qubit_content[q]

            if content is None:
                # Empty wire
                segment = '─' * width
            elif content[0] == 'gate':
                # Gate box
                symbol = content[1]
                padding = width - len(symbol) - 2
                left_pad = padding // 2
                right_pad = padding - left_pad
                segment = '─' * left_pad + f'[{symbol}]' + '─' * right_pad
            elif content[0] == 'control':
                # Control dot
                padding = width - 1
                left_pad = padding // 2
                right_pad = padding - left_pad
                segment = '─' * left_pad + '●' + '─' * right_pad

            lines[q].append(segment)

        # Draw vertical connectors
        for q in range(self.num_qubits):
            connector = ' ' * width

            # Check if this qubit needs a vertical line
            for (min_q, max_q, ctrls, tgt) in connections:
                if min_q < q < max_q:
                    # This qubit is between control and target
                    mid = width // 2
                    connector = ' ' * mid + '│' + ' ' * (width - mid - 1)
                    break

            connector_lines[q].append(connector)

    def _assemble_output(
        self,
        lines: Dict[int, List[str]],
        connector_lines: Dict[int, List[str]]
    ) -> str:
        """Assemble the final output string."""
        output = []

        # Calculate label width
        label_width = len(f"q{self.num_qubits - 1}: ")

        for q in range(self.num_qubits):
            # Qubit label
            label = f"q{q}: ".rjust(label_width)

            # Wire line
            wire = ''.join(lines[q])
            output.append(f"{label}{wire}")

            # Connector line (if not last qubit)
            if q < self.num_qubits - 1:
                connector = ''.join(connector_lines[q])
                if '│' in connector:
                    output.append(' ' * label_width + connector)

        return '\n'.join(output)


# =============================================================================
# Matplotlib Circuit Drawing
# =============================================================================

class MatplotlibCircuitDrawer:
    """
    Draws quantum circuits using matplotlib for image output.
    """

    # Colors
    GATE_COLOR = '#4A90D9'
    CONTROL_COLOR = '#2C3E50'
    WIRE_COLOR = '#7F8C8D'
    TEXT_COLOR = 'white'
    MEASURE_COLOR = '#E74C3C'

    # Gate display info: (symbol, width_factor)
    GATE_INFO = {
        'H': ('H', 1),
        'X': ('X', 1),
        'Y': ('Y', 1),
        'Z': ('Z', 1),
        'S': ('S', 1),
        'Sdg': ('S†', 1),
        'T': ('T', 1),
        'Tdg': ('T†', 1),
        'SX': ('√X', 1.2),
        'I': ('I', 1),
        'Rx': ('Rx', 1.2),
        'Ry': ('Ry', 1.2),
        'Rz': ('Rz', 1.2),
        'P': ('P', 1),
        'U': ('U', 1.2),
        'MCM': ('M', 1),
        'MEASURE': ('M', 1),
    }

    def __init__(self, circuit: 'Circuit'):
        """
        Initialize the matplotlib drawer.

        Args:
            circuit: The Circuit object to draw.
        """
        self.circuit = circuit
        self.num_qubits = circuit.size

    def draw(
        self,
        figsize: Optional[Tuple[float, float]] = None,
        show_angles: bool = True,
        title: Optional[str] = None,
        dpi: int = 150
    ) -> 'matplotlib.figure.Figure':
        """
        Generate a matplotlib figure of the circuit.

        Args:
            figsize: Figure size (width, height) in inches.
            show_angles: Whether to display angle parameters.
            title: Optional title for the figure.
            dpi: Resolution for the figure.

        Returns:
            matplotlib Figure object.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            raise ImportError(
                "matplotlib is required for circuit visualization. "
                "Install with: pip install matplotlib"
            )

        # Build layers
        layers = self._build_layers()
        num_layers = max(len(layers), 1)

        # Calculate figure size
        if figsize is None:
            width = max(8, num_layers * 1.2 + 2)
            height = max(4, self.num_qubits * 0.8 + 1)
            figsize = (width, height)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Draw wires
        for q in range(self.num_qubits):
            y = self.num_qubits - 1 - q
            ax.plot([-0.5, num_layers + 0.5], [y, y],
                   color=self.WIRE_COLOR, linewidth=1.5, zorder=1)
            ax.text(-0.8, y, f'q{q}', ha='right', va='center',
                   fontsize=12, fontweight='bold')

        # Draw gates layer by layer
        for layer_idx, layer in enumerate(layers):
            x = layer_idx + 0.5
            self._draw_layer(ax, layer, x, show_angles)

        # Configure axes
        ax.set_xlim(-1.5, num_layers + 1)
        ax.set_ylim(-0.8, self.num_qubits - 0.2)
        ax.set_aspect('equal')
        ax.axis('off')

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        return fig

    def _build_layers(self) -> List[List[Dict[str, Any]]]:
        """Organize gates into layers (same as ASCII drawer)."""
        layers: List[List[Dict[str, Any]]] = []
        qubit_depth = [0] * self.num_qubits

        for gate in self.circuit.circuit:
            if gate['name'] == 'BARRIER':
                max_depth = max(qubit_depth)
                qubit_depth = [max_depth] * self.num_qubits
                continue

            target = gate['target']
            controls = gate.get('controls', [])
            involved = [target] + controls

            layer_idx = max(qubit_depth[q] for q in involved)

            while len(layers) <= layer_idx:
                layers.append([])

            layers[layer_idx].append(gate)

            for q in involved:
                qubit_depth[q] = layer_idx + 1

        return layers

    def _draw_layer(
        self,
        ax,
        layer: List[Dict],
        x: float,
        show_angles: bool
    ) -> None:
        """Draw a single layer of gates."""
        import matplotlib.patches as patches

        for gate in layer:
            target = gate['target']
            controls = gate.get('controls', [])
            name = gate['name']

            y_target = self.num_qubits - 1 - target

            # Draw control connections
            if controls:
                for c in controls:
                    y_control = self.num_qubits - 1 - c

                    # Vertical line
                    ax.plot([x, x], [y_control, y_target],
                           color=self.CONTROL_COLOR, linewidth=2, zorder=2)

                    # Control dot
                    ax.scatter([x], [y_control], s=80, c=self.CONTROL_COLOR, zorder=3)

            # Handle SWAP
            if name == 'SWAP' and controls:
                y_control = self.num_qubits - 1 - controls[0]
                self._draw_swap_symbol(ax, x, y_target)
                self._draw_swap_symbol(ax, x, y_control)
                continue

            # Handle CNOT target
            if name in ('CNOT', 'CX'):
                self._draw_cnot_target(ax, x, y_target)
                continue

            # Handle CZ
            if name == 'CZ':
                ax.scatter([x], [y_target], s=80, c=self.CONTROL_COLOR, zorder=3)
                continue

            # Get gate info
            info = self.GATE_INFO.get(name, (name[:2], 1))
            symbol, width_factor = info

            # Add angle to symbol if present
            if show_angles and 'angles' in gate:
                angles = gate['angles']
                if isinstance(angles, (list, tuple)):
                    angle_str = ','.join(f'{a:.1f}' for a in angles)
                else:
                    angle_str = f'{angles:.1f}'
                symbol = f'{symbol}\n({angle_str})'
                width_factor = max(width_factor, 1.5)

            # Choose color
            color = self.MEASURE_COLOR if name in ('MCM', 'MEASURE') else self.GATE_COLOR

            # Draw gate box
            self._draw_gate_box(ax, x, y_target, symbol, width_factor, color)

    def _draw_gate_box(
        self,
        ax,
        x: float,
        y: float,
        text: str,
        width_factor: float,
        color: str
    ) -> None:
        """Draw a gate box with text."""
        import matplotlib.patches as patches

        width = 0.6 * width_factor
        height = 0.6

        rect = patches.FancyBboxPatch(
            (x - width/2, y - height/2),
            width, height,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=1.5,
            zorder=4
        )
        ax.add_patch(rect)

        ax.text(x, y, text, ha='center', va='center',
               fontsize=10, fontweight='bold', color=self.TEXT_COLOR, zorder=5)

    def _draw_cnot_target(self, ax, x: float, y: float) -> None:
        """Draw CNOT target (circled plus)."""
        circle = plt.Circle((x, y), 0.2, fill=False,
                           color=self.CONTROL_COLOR, linewidth=2, zorder=4)
        ax.add_patch(circle)
        ax.plot([x - 0.2, x + 0.2], [y, y], color=self.CONTROL_COLOR, linewidth=2, zorder=4)
        ax.plot([x, x], [y - 0.2, y + 0.2], color=self.CONTROL_COLOR, linewidth=2, zorder=4)

    def _draw_swap_symbol(self, ax, x: float, y: float) -> None:
        """Draw SWAP symbol (×)."""
        size = 0.15
        ax.plot([x - size, x + size], [y - size, y + size],
               color=self.CONTROL_COLOR, linewidth=2, zorder=4)
        ax.plot([x - size, x + size], [y + size, y - size],
               color=self.CONTROL_COLOR, linewidth=2, zorder=4)

    def save(
        self,
        filename: str,
        show_angles: bool = True,
        title: Optional[str] = None,
        dpi: int = 150,
        **kwargs
    ) -> None:
        """
        Save the circuit diagram to a file.

        Args:
            filename: Output filename (supports .png, .pdf, .svg).
            show_angles: Whether to display angle parameters.
            title: Optional title for the figure.
            dpi: Resolution for raster formats.
            **kwargs: Additional arguments passed to savefig.
        """
        fig = self.draw(show_angles=show_angles, title=title, dpi=dpi)
        fig.savefig(filename, dpi=dpi, bbox_inches='tight', **kwargs)
        plt.close(fig)
        print(f"Circuit saved to {filename}")


# =============================================================================
# Probability Visualization
# =============================================================================

class ProbabilityVisualizer:
    """
    Visualizes probability distributions and measurement results.
    """

    def __init__(self, circuit: 'Circuit'):
        """
        Initialize the probability visualizer.

        Args:
            circuit: The Circuit object to visualize.
        """
        self.circuit = circuit

    def print_probabilities(
        self,
        threshold: float = 1e-4,
        max_states: int = 32,
        sort_by: str = 'probability'
    ) -> None:
        """
        Print probability distribution to terminal.

        Args:
            threshold: Minimum probability to display.
            sort_by: 'probability' (descending) or 'state' (lexicographic).
        """
        if self.circuit.states is None:
            print("Circuit has not been executed. Call execute() first.")
            return

        probs = self.circuit.states.abs().pow(2).cpu().numpy()
        num_qubits = self.circuit.size

        # Build list of (state, probability)
        state_probs = []
        for i, p in enumerate(probs):
            if p >= threshold:
                state = format(i, f'0{num_qubits}b')[::-1]  # Little-endian
                state_probs.append((state, float(p)))

        # Sort
        if sort_by == 'probability':
            state_probs.sort(key=lambda x: -x[1])
        else:
            state_probs.sort(key=lambda x: x[0])

        # Truncate if needed
        truncated = len(state_probs) > max_states
        state_probs = state_probs[:max_states]

        # Print header
        print("\n" + "=" * 50)
        print("Probability Distribution")
        print("=" * 50)

        # Find max probability for bar scaling
        max_prob = max(p for _, p in state_probs) if state_probs else 0
        bar_width = 30

        for state, prob in state_probs:
            bar_len = int(prob / max_prob * bar_width) if max_prob > 0 else 0
            bar = '█' * bar_len + '░' * (bar_width - bar_len)
            print(f"  |{state}⟩  {bar}  {prob:.4f}")

        if truncated:
            print(f"  ... and more states below threshold or limit")

        print("=" * 50 + "\n")

    def plot_probabilities(
        self,
        figsize: Tuple[float, float] = (10, 6),
        threshold: float = 1e-4,
        max_states: int = 32,
        title: str = "Probability Distribution",
        color: str = '#4A90D9'
    ) -> 'matplotlib.figure.Figure':
        """
        Plot probability distribution as a bar chart.

        Args:
            figsize: Figure size.
            threshold: Minimum probability to display.
            max_states: Maximum states to show.
            title: Plot title.
            color: Bar color.

        Returns:
            matplotlib Figure object.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required. Install with: pip install matplotlib")

        if self.circuit.states is None:
            raise ValueError("Circuit has not been executed. Call execute() first.")

        probs = self.circuit.states.abs().pow(2).cpu().numpy()
        num_qubits = self.circuit.size

        # Build data
        states = []
        probabilities = []
        for i, p in enumerate(probs):
            if p >= threshold:
                state = format(i, f'0{num_qubits}b')[::-1]
                states.append(f"|{state}⟩")
                probabilities.append(float(p))

        # Sort by probability
        sorted_data = sorted(zip(states, probabilities), key=lambda x: -x[1])
        sorted_data = sorted_data[:max_states]
        states, probabilities = zip(*sorted_data) if sorted_data else ([], [])

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(states, probabilities, color=color, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Basis State', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(probabilities) * 1.1 if probabilities else 1)

        # Rotate labels if many states
        if len(states) > 8:
            plt.xticks(rotation=45, ha='right')

        # Add probability values on bars
        for bar, prob in zip(bars, probabilities):
            if prob > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{prob:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        return fig

    def plot_counts(
        self,
        figsize: Tuple[float, float] = (10, 6),
        title: str = "Measurement Results",
        color: str = '#2ECC71'
    ) -> 'matplotlib.figure.Figure':
        """
        Plot measurement counts as a bar chart.

        Args:
            figsize: Figure size.
            title: Plot title.
            color: Bar color.

        Returns:
            matplotlib Figure object.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required. Install with: pip install matplotlib")

        counts = self.circuit.get_counts()
        if counts is None:
            raise ValueError("No measurement results. Call execute() first.")

        # Sort by state
        sorted_items = sorted(counts.items())
        states = [f"|{s}⟩" for s, _ in sorted_items]
        values = [c for _, c in sorted_items]

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(states, values, color=color, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Basis State', fontsize=12)
        ax.set_ylabel('Counts', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        if len(states) > 8:
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        return fig


# =============================================================================
# Main Visualization Interface
# =============================================================================

class CircuitVisualizer:
    """
    Unified interface for all circuit visualization methods.

    Example:
        >>> qc = Circuit(2).h(0).cnot(0, 1).execute(shots=1000)
        >>> vis = CircuitVisualizer(qc)
        >>> vis.print_circuit()           # ASCII to terminal
        >>> vis.draw_circuit()            # Matplotlib figure
        >>> vis.save_circuit("circ.png")  # Save to file
        >>> vis.print_probabilities()     # Probabilities to terminal
        >>> vis.plot_probabilities()      # Probability bar chart
    """

    def __init__(self, circuit: 'Circuit'):
        """
        Initialize the visualizer.

        Args:
            circuit: The Circuit object to visualize.
        """
        self.circuit = circuit
        self._ascii_drawer = ASCIICircuitDrawer(circuit)
        self._mpl_drawer = MatplotlibCircuitDrawer(circuit)
        self._prob_viz = ProbabilityVisualizer(circuit)

    # -------------------------------------------------------------------------
    # Circuit Visualization
    # -------------------------------------------------------------------------

    def print_circuit(self, show_angles: bool = True) -> None:
        """
        Print ASCII circuit diagram to terminal.

        Args:
            show_angles: Whether to display angle parameters.
        """
        print(self._ascii_drawer.draw(show_angles=show_angles))

    def draw_circuit(
        self,
        figsize: Optional[Tuple[float, float]] = None,
        show_angles: bool = True,
        title: Optional[str] = None,
        dpi: int = 150
    ) -> 'matplotlib.figure.Figure':
        """
        Draw circuit diagram using matplotlib.

        Args:
            figsize: Figure size (width, height) in inches.
            show_angles: Whether to display angle parameters.
            title: Optional title.
            dpi: Resolution.

        Returns:
            matplotlib Figure object.
        """
        return self._mpl_drawer.draw(figsize, show_angles, title, dpi)

    def save_circuit(
        self,
        filename: str,
        show_angles: bool = True,
        title: Optional[str] = None,
        dpi: int = 150,
        **kwargs
    ) -> None:
        """
        Save circuit diagram to file.

        Args:
            filename: Output filename (.png, .pdf, .svg).
            show_angles: Whether to display angle parameters.
            title: Optional title.
            dpi: Resolution for raster formats.
        """
        self._mpl_drawer.save(filename, show_angles, title, dpi, **kwargs)

    # -------------------------------------------------------------------------
    # Probability Visualization
    # -------------------------------------------------------------------------

    def print_probabilities(
        self,
        threshold: float = 1e-4,
        max_states: int = 32,
        sort_by: str = 'probability'
    ) -> None:
        """
        Print probability distribution to terminal.

        Args:
            threshold: Minimum probability to display.
            max_states: Maximum number of states to show.
            sort_by: 'probability' or 'state'.
        """
        self._prob_viz.print_probabilities(threshold, max_states, sort_by)

    def plot_probabilities(
        self,
        figsize: Tuple[float, float] = (10, 6),
        threshold: float = 1e-4,
        max_states: int = 32,
        title: str = "Probability Distribution",
        color: str = '#4A90D9'
    ) -> 'matplotlib.figure.Figure':
        """
        Plot probability distribution.

        Args:
            figsize: Figure size.
            threshold: Minimum probability to display.
            max_states: Maximum states to show.
            title: Plot title.
            color: Bar color.

        Returns:
            matplotlib Figure object.
        """
        return self._prob_viz.plot_probabilities(
            figsize, threshold, max_states, title, color
        )

    def plot_counts(
        self,
        figsize: Tuple[float, float] = (10, 6),
        title: str = "Measurement Results",
        color: str = '#2ECC71'
    ) -> 'matplotlib.figure.Figure':
        """
        Plot measurement counts.

        Args:
            figsize: Figure size.
            title: Plot title.
            color: Bar color.

        Returns:
            matplotlib Figure object.
        """
        return self._prob_viz.plot_counts(figsize, title, color)

    def save_probabilities(
        self,
        filename: str,
        figsize: Tuple[float, float] = (10, 6),
        threshold: float = 1e-4,
        max_states: int = 32,
        title: str = "Probability Distribution",
        dpi: int = 150
    ) -> None:
        """
        Save probability plot to file.

        Args:
            filename: Output filename.
            figsize: Figure size.
            threshold: Minimum probability to display.
            max_states: Maximum states to show.
            title: Plot title.
            dpi: Resolution.
        """
        import matplotlib.pyplot as plt
        fig = self.plot_probabilities(figsize, threshold, max_states, title)
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Probability plot saved to {filename}")

    def save_counts(
        self,
        filename: str,
        figsize: Tuple[float, float] = (10, 6),
        title: str = "Measurement Results",
        dpi: int = 150
    ) -> None:
        """
        Save measurement counts plot to file.

        Args:
            filename: Output filename.
            figsize: Figure size.
            title: Plot title.
            dpi: Resolution.
        """
        import matplotlib.pyplot as plt
        fig = self.plot_counts(figsize, title)
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Counts plot saved to {filename}")

    # -------------------------------------------------------------------------
    # Combined Visualization
    # -------------------------------------------------------------------------

    def summary(
        self,
        show_circuit: bool = True,
        show_probabilities: bool = True,
        show_counts: bool = True
    ) -> None:
        """
        Print a complete summary of the circuit and results.

        Args:
            show_circuit: Show ASCII circuit diagram.
            show_probabilities: Show probability distribution.
            show_counts: Show measurement counts.
        """
        print("\n" + "=" * 60)
        print("QUANTUM CIRCUIT SUMMARY")
        print("=" * 60)

        print(f"\nQubits: {self.circuit.size}")
        print(f"Gates: {len(self.circuit)}")
        print(f"Depth: {self.circuit.depth()}")
        print(f"Gate counts: {self.circuit.gate_count()}")

        if show_circuit:
            print("\n" + "-" * 60)
            print("CIRCUIT DIAGRAM")
            print("-" * 60)
            self.print_circuit()

        if show_probabilities and self.circuit.states is not None:
            self.print_probabilities()

        if show_counts and self.circuit.measurements is not None:
            counts = self.circuit.get_counts()
            if counts:
                print("-" * 60)
                print("MEASUREMENT COUNTS")
                print("-" * 60)
                total = sum(counts.values())
                for state, count in sorted(counts.items(), key=lambda x: -x[1]):
                    pct = count / total * 100
                    print(f"  |{state}⟩: {count} ({pct:.1f}%)")

        print("\n" + "=" * 60 + "\n")


# =============================================================================
# Convenience Functions
# =============================================================================

def print_circuit(circuit: 'Circuit', show_angles: bool = True) -> None:
    """
    Print ASCII circuit diagram to terminal.

    Args:
        circuit: The Circuit to visualize.
        show_angles: Whether to display angle parameters.
    """
    CircuitVisualizer(circuit).print_circuit(show_angles)


def draw_circuit(
    circuit: 'Circuit',
    figsize: Optional[Tuple[float, float]] = None,
    show_angles: bool = True,
    title: Optional[str] = None
) -> 'matplotlib.figure.Figure':
    """
    Draw circuit diagram using matplotlib.

    Args:
        circuit: The Circuit to visualize.
        figsize: Figure size.
        show_angles: Whether to display angle parameters.
        title: Optional title.

    Returns:
        matplotlib Figure object.
    """
    return CircuitVisualizer(circuit).draw_circuit(figsize, show_angles, title)


def save_circuit(
    circuit: 'Circuit',
    filename: str,
    show_angles: bool = True,
    title: Optional[str] = None,
    dpi: int = 150
) -> None:
    """
    Save circuit diagram to file.

    Args:
        circuit: The Circuit to visualize.
        filename: Output filename.
        show_angles: Whether to display angle parameters.
        title: Optional title.
        dpi: Resolution.
    """
    CircuitVisualizer(circuit).save_circuit(filename, show_angles, title, dpi)


def circuit_summary(circuit: 'Circuit') -> None:
    """
    Print a complete summary of the circuit and results.

    Args:
        circuit: The Circuit to summarize.
    """
    CircuitVisualizer(circuit).summary()
