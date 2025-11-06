import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
import time
from brain_snn_numpy import BrainSNN
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns
from typing import Dict, List, Optional, Tuple

class BrainSNNTelemetry:
    """
    Telemetry dashboard for monitoring and visualizing BrainSNN networks.
    """
    def __init__(self, snn: BrainSNN):
        """
        Initialize the telemetry dashboard with a BrainSNN instance.
        
        Args:
            snn: The BrainSNN instance to monitor
        """
        self.snn = snn
        self.history = {
            'membrane_potential': [],
            'has_spiked': [],
            'neuron_activity': [],
            'weights': [],
            'new_neurons': [],
            'new_connections': []
        }
        self.step_count = 0
        self.neuron_labels = {}
        self._initialize_neuron_labels()
        
    def _initialize_neuron_labels(self):
        """Create labels for each neuron based on its type."""
        for i in range(self.snn.num_neurons):
            if i in self.snn.input_indices:
                self.neuron_labels[i] = f"IN-{i}"
            elif i in self.snn.hidden_indices:
                idx = np.where(self.snn.hidden_indices == i)[0][0]
                self.neuron_labels[i] = f"H-{idx}"
            elif i in self.snn.output_indices:
                idx = np.where(self.snn.output_indices == i)[0][0]
                self.neuron_labels[i] = f"OUT-{idx}"
                
    def record_step(self):
        """Record the current state of the network."""
        # Pad history arrays if new neurons were added
        current_size = self.snn.num_neurons
        for key in ['membrane_potential', 'has_spiked', 'neuron_activity']:
            if self.history[key]:
                last_record = self.history[key][-1]
                if len(last_record) < current_size:
                    # Pad with zeros
                    padded_record = np.zeros(current_size, dtype=last_record.dtype)
                    padded_record[:len(last_record)] = last_record
                    self.history[key][-1] = padded_record
        
        # Record current state
        self.history['membrane_potential'].append(self.snn.membrane_potential.copy())
        self.history['has_spiked'].append(self.snn.has_spiked.copy())
        self.history['neuron_activity'].append(self.snn.neuron_activity.copy())
        self.history['weights'].append(self.snn.weights.copy())
        self.step_count += 1
        
    def record_plasticity(self, new_neurons: int, new_connections: int):
        """Record plasticity events."""
        self.history['new_neurons'].append(new_neurons)
        self.history['new_connections'].append(new_connections)
        if new_neurons > 0:
            # Update neuron labels if new neurons were added
            self._initialize_neuron_labels()
            
            # Pad all history arrays to match new network size
            current_size = self.snn.num_neurons
            for key in ['membrane_potential', 'has_spiked', 'neuron_activity']:
                if self.history[key]:
                    for i in range(len(self.history[key])):
                        last_record = self.history[key][i]
                        if len(last_record) < current_size:
                            # Pad with zeros
                            padded_record = np.zeros(current_size, dtype=last_record.dtype)
                            padded_record[:len(last_record)] = last_record
                            self.history[key][i] = padded_record
            
    def clear_history(self):
        """Clear the recorded history."""
        for key in self.history:
            self.history[key] = []
        self.step_count = 0
        
    def plot_network_graph(self, figsize=(12, 10), threshold=0.0):
        """
        Plot the network as a directed graph.
        
        Args:
            figsize: Figure size (width, height)
            threshold: Minimum absolute weight value to show a connection
        """
        G = nx.DiGraph()
        
        # Add nodes with positions
        input_positions = {}
        hidden_positions = {}
        output_positions = {}
        
        # Calculate positions for neat visualization
        n_inputs = len(self.snn.input_indices)
        n_hidden = len(self.snn.hidden_indices)
        n_outputs = len(self.snn.output_indices)
        
        # Set positions for each neuron type
        for i, idx in enumerate(self.snn.input_indices):
            y_pos = (i - n_inputs/2) / max(1, n_inputs-1) * 0.8
            input_positions[idx] = (-1, y_pos)
            G.add_node(idx, color='blue', label=self.neuron_labels[idx], type='input')
            
        for i, idx in enumerate(self.snn.hidden_indices):
            x_offset = (i % 3 - 1) * 0.3  # Distribute in 3 columns for better visibility
            layer = i // 3
            max_layers = n_hidden // 3 + 1
            y_pos = ((i // 3) - max_layers/2) / max(1, max_layers-1) * 0.8
            hidden_positions[idx] = (x_offset, y_pos)
            G.add_node(idx, color='green', label=self.neuron_labels[idx], type='hidden')
            
        for i, idx in enumerate(self.snn.output_indices):
            y_pos = (i - n_outputs/2) / max(1, n_outputs-1) * 0.8
            output_positions[idx] = (1, y_pos)
            G.add_node(idx, color='red', label=self.neuron_labels[idx], type='output')
            
        # Combine all positions
        pos = {**input_positions, **hidden_positions, **output_positions}
        
        # Add edges with weights as attributes
        weights = self.snn.weights
        for i in range(self.snn.num_neurons):
            for j in range(self.snn.num_neurons):
                if abs(weights[i, j]) > threshold:
                    G.add_edge(i, j, weight=weights[i, j], 
                               width=1.0 + 3.0 * abs(weights[i, j]))
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw nodes by type with different colors
        node_colors = [data['color'] for _, data in G.nodes(data=True)]
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=ax, 
                               node_size=500, alpha=0.8)
        
        # Draw edges with color based on weight
        pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > 0]
        neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < 0]
        
        edge_widths_pos = [G[u][v]['width'] for u, v in pos_edges]
        edge_widths_neg = [G[u][v]['width'] for u, v in neg_edges]
        
        nx.draw_networkx_edges(G, pos, edgelist=pos_edges, edge_color='green', 
                               width=edge_widths_pos, alpha=0.6, ax=ax, 
                               connectionstyle='arc3,rad=0.1')
        nx.draw_networkx_edges(G, pos, edgelist=neg_edges, edge_color='red', 
                               width=edge_widths_neg, alpha=0.6, ax=ax, 
                               connectionstyle='arc3,rad=0.1')
        
        # Draw node labels
        labels = {n: data['label'] for n, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold', ax=ax)
        
        # Add a title
        plt.title(f"Network Graph - {self.snn.num_inputs} inputs, {self.snn.num_hidden_neurons} hidden, {self.snn.num_outputs} outputs")
        plt.axis('off')
        
        # Add legend
        legend_elements = [
            Patch(facecolor='blue', label='Input Neurons'),
            Patch(facecolor='green', label='Hidden Neurons'),
            Patch(facecolor='red', label='Output Neurons')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        return fig
        
    def plot_membrane_potentials(self, figsize=(12, 8)):
        """Plot the membrane potentials of all neurons as a bar graph."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get current membrane potentials
        potentials = self.snn.membrane_potential
        
        # Create neuron indices for x-axis
        indices = np.arange(self.snn.num_neurons)
        
        # Create masks for each neuron type
        input_mask = np.isin(indices, self.snn.input_indices)
        hidden_mask = np.isin(indices, self.snn.hidden_indices)
        output_mask = np.isin(indices, self.snn.output_indices)
        
        # Create bars for each neuron type with different colors
        bar_width = 0.8
        
        # Input neurons (blue)
        ax.bar(indices[input_mask], potentials[input_mask], 
               width=bar_width, color='blue', alpha=0.7, label='Input Neurons')
        
        # Hidden neurons (green)
        ax.bar(indices[hidden_mask], potentials[hidden_mask], 
               width=bar_width, color='green', alpha=0.7, label='Hidden Neurons')
        
        # Output neurons (red)
        ax.bar(indices[output_mask], potentials[output_mask], 
               width=bar_width, color='red', alpha=0.7, label='Output Neurons')
        
        # Plot the threshold line
        ax.axhline(y=self.snn.threshold[0], color='black', linestyle='--', 
                   label=f"Threshold: {self.snn.threshold[0]:.2f}")
        
        # Set custom x-tick labels
        x_labels = []
        for i in range(self.snn.num_neurons):
            if i in self.snn.input_indices:
                idx = np.where(self.snn.input_indices == i)[0][0]
                x_labels.append(f"IN-{idx}")
            elif i in self.snn.hidden_indices:
                idx = np.where(self.snn.hidden_indices == i)[0][0]
                x_labels.append(f"H-{idx}")
            elif i in self.snn.output_indices:
                idx = np.where(self.snn.output_indices == i)[0][0]
                x_labels.append(f"OUT-{idx}")
        
        plt.xticks(indices, x_labels, rotation=90, fontsize=8)
        
        plt.title("Current Membrane Potentials")
        plt.xlabel("Neuron")
        plt.ylabel("Membrane Potential")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        return fig
    
    def plot_spikes(self, figsize=(12, 8)):
        """Plot spike raster for all neurons."""
        fig, ax = plt.subplots(figsize=figsize)
        
        if not self.history['has_spiked']:
            plt.title("No spike data recorded")
            return fig
            
        spikes = np.array(self.history['has_spiked'])
        
        # Filter neurons that have spiked at least once
        active_neurons = []
        neuron_types = []
        neuron_labels = []
        
        for idx in range(self.snn.num_neurons):
            if np.any(spikes[:, idx]):
                active_neurons.append(idx)
                if idx in self.snn.input_indices:
                    neuron_types.append('input')
                    i = np.where(self.snn.input_indices == idx)[0][0]
                    neuron_labels.append(f"IN-{i}")
                elif idx in self.snn.hidden_indices:
                    neuron_types.append('hidden')
                    i = np.where(self.snn.hidden_indices == idx)[0][0]
                    neuron_labels.append(f"H-{i}")
                elif idx in self.snn.output_indices:
                    neuron_types.append('output')
                    i = np.where(self.snn.output_indices == idx)[0][0]
                    neuron_labels.append(f"OUT-{i}")
        
        # Plot spikes
        for i, neuron_idx in enumerate(active_neurons):
            spike_times = np.where(spikes[:, neuron_idx])[0]
            if neuron_types[i] == 'input':
                color = 'blue'
            elif neuron_types[i] == 'hidden':
                color = 'green'
            else:
                color = 'red'
            
            ax.scatter(spike_times, [i] * len(spike_times), marker='|', s=100, 
                      color=color, label=neuron_labels[i] if i < 10 else "")
        
        # Set y-ticks to show neuron labels
        if active_neurons:
            plt.yticks(range(len(active_neurons)), neuron_labels)
        
        plt.title("Spike Raster Plot")
        plt.xlabel("Time Steps")
        plt.ylabel("Neuron")
        
        # Create a custom legend for neuron types
        handles = [
            Patch(color='blue', label='Input Neurons'),
            Patch(color='green', label='Hidden Neurons'),
            Patch(color='red', label='Output Neurons')
        ]
        plt.legend(handles=handles, loc='upper right')
        
        plt.grid(True, alpha=0.3, axis='x')
        
        return fig
    
    def plot_neuron_activity(self, figsize=(12, 8)):
        """Plot neuron activity over time."""
        fig, ax = plt.subplots(figsize=figsize)
        
        if not self.history['neuron_activity']:
            plt.title("No neuron activity data recorded")
            return fig
            
        activity = np.array(self.history['neuron_activity'])
        time_steps = np.arange(activity.shape[0])
        
        # Plot input neurons
        for i, idx in enumerate(self.snn.input_indices):
            if idx < activity.shape[1]:  # Check if index exists in history
                ax.plot(time_steps, activity[:, idx], 
                        color='blue', alpha=0.7, 
                        label=f"IN-{i}" if i == 0 else "")
                
        # Plot hidden neurons
        for i, idx in enumerate(self.snn.hidden_indices):
            if idx < activity.shape[1]:  # Check if index exists in history
                ax.plot(time_steps, activity[:, idx], 
                        color='green', alpha=0.7, 
                        label=f"H-{i}" if i == 0 else "")
                
        # Plot output neurons
        for i, idx in enumerate(self.snn.output_indices):
            if idx < activity.shape[1]:  # Check if index exists in history
                ax.plot(time_steps, activity[:, idx], 
                        color='red', alpha=0.7, 
                        label=f"OUT-{i}" if i == 0 else "")
        
        # Plot the plasticity threshold line
        ax.axhline(y=self.snn.plasticity_threshold, color='purple', linestyle='--', 
                   label=f"Plasticity Threshold: {self.snn.plasticity_threshold:.2f}")
        
        plt.title("Neuron Activity Over Time")
        plt.xlabel("Time Steps")
        plt.ylabel("Activity Level")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        
        return fig
    
    def plot_weight_matrix(self, figsize=(10, 8)):
        """Plot the weight matrix as a heatmap."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get current weights
        weights = self.snn.weights
        
        # Create a mask for zero weights (no connection)
        mask = weights == 0
        
        # Plot heatmap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(weights, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, 
                    center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .5}, ax=ax)
        
        # Set labels for axes
        n_inputs = len(self.snn.input_indices)
        n_hidden = len(self.snn.hidden_indices)
        n_outputs = len(self.snn.output_indices)
        
        # Add dividing lines between neuron types
        ax.axhline(y=n_inputs, color='black', linestyle='-', linewidth=2)
        ax.axhline(y=n_inputs + n_hidden, color='black', linestyle='-', linewidth=2)
        ax.axvline(x=n_inputs, color='black', linestyle='-', linewidth=2)
        ax.axvline(x=n_inputs + n_hidden, color='black', linestyle='-', linewidth=2)
        
        # Add labels
        plt.title("Weight Matrix")
        plt.xlabel("Target Neuron")
        plt.ylabel("Source Neuron")
        
        # Add text labels to identify neuron types
        ax.text(n_inputs/2, -0.5, "Inputs", ha='center', va='top', fontsize=12)
        ax.text(n_inputs + n_hidden/2, -0.5, "Hidden", ha='center', va='top', fontsize=12)
        ax.text(n_inputs + n_hidden + n_outputs/2, -0.5, "Outputs", ha='center', va='top', fontsize=12)
        
        ax.text(-0.5, n_inputs/2, "Inputs", ha='right', va='center', fontsize=12, rotation=90)
        ax.text(-0.5, n_inputs + n_hidden/2, "Hidden", ha='right', va='center', fontsize=12, rotation=90)
        ax.text(-0.5, n_inputs + n_hidden + n_outputs/2, "Outputs", ha='right', va='center', fontsize=12, rotation=90)
        
        return fig
    
    def plot_plasticity_history(self, figsize=(10, 6)):
        """Plot the history of plasticity events."""
        fig, ax = plt.subplots(figsize=figsize)
        
        if not self.history['new_neurons']:
            plt.title("No plasticity data recorded")
            return fig
            
        time_steps = np.arange(len(self.history['new_neurons']))
        
        ax.bar(time_steps, self.history['new_neurons'], color='green', 
               alpha=0.7, label='New Neurons')
        ax.bar(time_steps, self.history['new_connections'], color='blue', 
               alpha=0.7, label='New Connections')
        
        plt.title("Plasticity Events Over Time")
        plt.xlabel("Time Steps")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return fig
    
    def plot_plasticity_parameters(self, figsize=(10, 6)):
        """Plot the current plasticity parameters."""
        params = self.snn.get_plasticity_params()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create a bar plot for the parameters
        param_names = list(params.keys())
        param_values = list(params.values())
        
        ax.bar(param_names, param_values, color='purple', alpha=0.7)
        
        plt.title("Plasticity Parameters")
        plt.ylabel("Value")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        return fig

# Streamlit Dashboard
def run_dashboard():
    st.set_page_config(page_title="BrainSNN Telemetry Dashboard", layout="wide")
    
    st.title("BrainSNN Telemetry Dashboard")
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.snn = None
        st.session_state.telemetry = None
        st.session_state.input_spikes = []
        st.session_state.step_count = 0
        st.session_state.run_simulation = False
    
    # Sidebar for network configuration
    with st.sidebar:
        st.header("Network Configuration")
        
        num_inputs = st.slider("Number of Input Neurons", 1, 20, 4)
        num_hidden = st.slider("Number of Hidden Neurons", 1, 50, 10)
        num_outputs = st.slider("Number of Output Neurons", 1, 10, 2)
        
        threshold = st.slider("Neuron Threshold", 0.1, 5.0, 1.0, 0.1)
        decay_rate = st.slider("Membrane Potential Decay Rate", 0.0, 1.0, 0.2, 0.05)
        connection_prob = st.slider("Connection Probability", 0.0, 1.0, 0.3, 0.05)
        
        st.subheader("Plasticity Parameters")
        neuron_growth_rate = st.slider("Neuron Growth Rate", 0.0, 0.1, 0.01, 0.005)
        connection_growth_rate = st.slider("Connection Growth Rate", 0.0, 0.2, 0.05, 0.01)
        plasticity_threshold = st.slider("Plasticity Threshold", 0.0, 2.0, 0.7, 0.1)
        
        if st.button("Initialize Network"):
            with st.spinner("Initializing network..."):
                # Create a new BrainSNN instance
                st.session_state.snn = BrainSNN(
                    num_inputs=num_inputs,
                    num_outputs=num_outputs,
                    num_hidden_neurons=num_hidden,
                    threshold=threshold,
                    decay_rate=decay_rate,
                    connection_probability=connection_prob,
                    neuron_growth_rate=neuron_growth_rate,
                    connection_growth_rate=connection_growth_rate,
                    plasticity_threshold=plasticity_threshold
                )
                
                # Create telemetry
                st.session_state.telemetry = BrainSNNTelemetry(st.session_state.snn)
                st.session_state.initialized = True
                st.session_state.step_count = 0
                st.session_state.input_spikes = [False] * num_inputs
                st.success("Network initialized successfully!")
    
    # Main dashboard
    if not st.session_state.initialized:
        st.info("Please initialize the network using the sidebar controls.")
    else:
        # Input controls
        st.header("Input Control")
        
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            # Create toggles for input neurons
            st.write("Input Neurons (toggle to activate):")
            input_cols = st.columns(min(8, st.session_state.snn.num_inputs))
            new_input_spikes = list(st.session_state.input_spikes)
            
            for i in range(st.session_state.snn.num_inputs):
                col_idx = i % len(input_cols)
                with input_cols[col_idx]:
                    new_input_spikes[i] = st.toggle(f"IN-{i}", value=new_input_spikes[i])
            
            st.session_state.input_spikes = new_input_spikes
        
        with col2:
            step_button = st.button("Step Forward")
        
        with col3:
            apply_plasticity = st.button("Apply Plasticity")
        
        with col4:
            reset_button = st.button("Reset Network")
        
        # Auto-run controls
        auto_run = st.checkbox("Auto Run Simulation", value=st.session_state.run_simulation)
        st.session_state.run_simulation = auto_run
        
        if auto_run:
            interval = st.slider("Step Interval (seconds)", 0.1, 2.0, 0.5, 0.1)
        
        # Process button actions
        if step_button or (auto_run and time.time() % interval < 0.1):
            with st.spinner("Processing step..."):
                # Forward pass
                output_spikes = st.session_state.snn.forward(st.session_state.input_spikes)
                # Record state
                st.session_state.telemetry.record_step()
                st.session_state.step_count += 1
        
        if apply_plasticity:
            with st.spinner("Applying plasticity..."):
                new_neurons, new_connections = st.session_state.snn.apply_plasticity()
                st.session_state.telemetry.record_plasticity(new_neurons, new_connections)
                st.success(f"Plasticity applied: {new_neurons} new neurons, {new_connections} new connections")
        
        if reset_button:
            with st.spinner("Resetting network..."):
                st.session_state.snn.reset()
                st.session_state.telemetry.clear_history()
                st.session_state.step_count = 0
                st.session_state.input_spikes = [False] * st.session_state.snn.num_inputs
                st.success("Network reset successfully!")
        
        # Network statistics
        st.header("Network Statistics")
        
        stat_cols = st.columns(5)
        with stat_cols[0]:
            st.metric("Step Count", st.session_state.step_count)
        with stat_cols[1]:
            st.metric("# Input Neurons", st.session_state.snn.num_inputs)
        with stat_cols[2]:
            st.metric("# Hidden Neurons", st.session_state.snn.num_hidden_neurons)
        with stat_cols[3]:
            st.metric("# Output Neurons", st.session_state.snn.num_outputs)
        with stat_cols[4]:
            # Count non-zero weights to get connection count
            connection_count = np.count_nonzero(st.session_state.snn.weights)
            st.metric("# Connections", connection_count)
        
        # Visualizations
        st.header("Network Visualizations")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Network Graph", "Membrane Potentials", "Spike Activity", 
            "Neuron Activity", "Weight Matrix"
        ])
        
        with tab1:
            st.subheader("Network Graph")
            weight_threshold = st.slider("Connection Weight Threshold", 0.0, 1.0, 0.0, 0.05)
            network_fig = st.session_state.telemetry.plot_network_graph(threshold=weight_threshold)
            st.pyplot(network_fig)
        
        with tab2:
            st.subheader("Membrane Potentials")
            potential_fig = st.session_state.telemetry.plot_membrane_potentials()
            st.pyplot(potential_fig)
        
        with tab3:
            st.subheader("Spike Activity")
            spike_fig = st.session_state.telemetry.plot_spikes()
            st.pyplot(spike_fig)
        
        with tab4:
            st.subheader("Neuron Activity")
            activity_fig = st.session_state.telemetry.plot_neuron_activity()
            st.pyplot(activity_fig)
        
        with tab5:
            st.subheader("Weight Matrix")
            weight_fig = st.session_state.telemetry.plot_weight_matrix()
            st.pyplot(weight_fig)

if __name__ == "__main__":
    run_dashboard() 