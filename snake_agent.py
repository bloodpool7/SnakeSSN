import numpy as np
from typing import Dict, List, Tuple, Optional
from brain_snn_numpy import BrainSNN

class SnakeAgent:
    def __init__(
        self,
        num_vision_rays: int = 8,
        num_hidden_neurons: int = 50,
        max_hidden_neurons: int = 200,
        latency_window: int = 20,
        connection_probability: float = 0.4,
        plasticity_threshold: float = 0.5,
        neuron_growth_rate: float = 0.01,
        connection_growth_rate: float = 0.05
    ):
        """
        Initialize a Snake agent using a BrainSNN with latency-based decoding.
        
        Args:
            num_vision_rays: Number of vision rays for sensing
            num_hidden_neurons: Initial number of hidden neurons
            max_hidden_neurons: Maximum number of hidden neurons
            latency_window: Maximum time steps to wait for output spikes
            connection_probability: Initial connection probability
            plasticity_threshold: Threshold for plasticity mechanisms
            neuron_growth_rate: Rate of neuron growth
            connection_growth_rate: Rate of connection growth
        """
        # Calculate input and output dimensions
        # Inputs: vision rays (8×3), direction (4), danger (3), food direction (2)
        num_inputs = (num_vision_rays * 3) + 4 + 3 + 2
        num_outputs = 4  # UP, RIGHT, DOWN, LEFT
        
        # Create BrainSNN
        self.snn = BrainSNN(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            num_hidden_neurons=num_hidden_neurons,
            max_hidden_neurons=max_hidden_neurons,
            connection_probability=connection_probability,
            plasticity_threshold=plasticity_threshold,
            neuron_growth_rate=neuron_growth_rate,
            connection_growth_rate=connection_growth_rate
        )
        
        # Latency decoding parameters
        self.latency_window = latency_window
        self.current_direction = 1  # Start facing RIGHT
        
        # For storing performance metrics
        self.episode_scores = []
        self.last_reward = 0
        self.cumulative_reward = 0
        
        # Store agent parameters
        self.num_vision_rays = num_vision_rays
        self.using_latency_coding = True
    
    def process_state(self, state: Dict) -> np.ndarray:
        """
        Convert state dictionary to input spikes for BrainSNN.
        
        Args:
            state: Dictionary containing state information
                - direction: One-hot encoded current direction
                - danger: Binary array for forward, right, left danger
                - vision: Flattened vision ray information
                - food_direction: Normalized vector pointing to food
        
        Returns:
            input_spikes: Binary array of input spikes
        """
        # Get components from state
        direction = state["direction"]
        danger = state["danger"]
        vision = state["vision"]
        food_direction = state["food_direction"]
        
        # Normalize all inputs to [0,1] range for conversion to spikes
        # Direction and danger are already binary
        
        # Combine all inputs into a single vector
        combined_input = np.concatenate([vision, direction, danger, food_direction])
        
        # Convert to binary spikes using threshold crossing
        # Higher values = more likely to spike
        # Use vectorized random sampling
        spike_probabilities = combined_input
        input_spikes = self.rng.random(size=len(combined_input)) < spike_probabilities
        
        return input_spikes
    
    def decide_action_latency(self, state: Dict) -> int:
        """
        Decide action using latency coding.
        
        Args:
            state: Dictionary containing state information
        
        Returns:
            action: Integer representing the chosen action (0-3)
        """
        # Process state to get input spikes
        input_spikes = self.process_state(state)
        
        # Reset membrane potentials for a fresh decision
        self.snn.reset()
        
        # Initialize latency tracking
        first_spike_time = np.ones(4) * (self.latency_window + 1)  # Init to max+1 (no spike)
        
        # Process network for multiple time steps
        for t in range(self.latency_window):
            outputs = self.snn.forward(input_spikes)
            
            # For each output that hasn't spiked yet but spikes now, record the time
            new_spikes = np.logical_and(outputs, first_spike_time > self.latency_window)
            first_spike_mask = np.where(new_spikes)[0]
            
            if len(first_spike_mask) > 0:
                for idx in first_spike_mask:
                    first_spike_time[idx] = t
            
            # If direction opposite to current isn't allowed, make it spike very late
            opposite_dir = (self.current_direction + 2) % 4
            first_spike_time[opposite_dir] = self.latency_window + 1
            
            # Optional early stopping if at least one direction has been chosen
            if np.any(first_spike_time <= self.latency_window):
                break
        
        # If any valid spikes occurred, take the earliest one
        if np.any(first_spike_time <= self.latency_window):
            action = np.argmin(first_spike_time)
        else:
            # No decision - maintain current direction
            action = self.current_direction
        
        # Update current direction
        self.current_direction = action
        
        return action
    
    def act(self, state: Dict) -> int:
        """
        Choose an action based on the current state.
        
        Args:
            state: Dictionary containing state information
            
        Returns:
            action: Integer representing the chosen action (0-3)
        """
        # Using latency-based decoding
        return self.decide_action_latency(state)
    
    def update(self, reward: float) -> None:
        """
        Update the agent based on the received reward.
        
        Args:
            reward: Reward signal from the environment
        """
        # Store reward information
        self.last_reward = reward
        self.cumulative_reward += reward
        
        # Apply plasticity based on reward
        if reward > 0:
            # Positive reward - apply normal plasticity
            self.snn.apply_plasticity()
        elif reward < 0:
            # Negative reward - no plasticity but reset activity
            # This prevents forming connections based on bad decisions
            self.snn.neuron_activity *= 0.5
    
    def mutate_parameters(self, mutation_rate: float = 0.1) -> None:
        """
        Mutate the agent's parameters and neural network weights.
        
        Args:
            mutation_rate: Rate of mutation
        """
        # 1. Mutate hyperparameters
        params = self.snn.get_plasticity_params()
        
        for key in params:
            if np.random.random() < mutation_rate:
                # Add random noise based on parameter value
                mutation = np.random.normal(0, params[key] * 0.2)
                params[key] = max(0.001, params[key] + mutation)
        
        # Set mutated parameters
        self.snn.set_plasticity_params(params)
        
        # 2. Mutate neural network weights
        # Get existing weights
        weights = self.snn.weights
        
        # Create a mutation mask (True where we want to mutate)
        mutation_mask = np.random.random(weights.shape) < mutation_rate
        
        # Only mutate existing connections (non-zero weights)
        existing_connections = weights != 0
        mutation_mask = np.logical_and(mutation_mask, existing_connections)
        
        # Generate random mutations (scaled by current weight values)
        weight_mutations = np.random.normal(0, 0.1, size=weights.shape)
        
        # Apply mutations only where the mask is True
        weights[mutation_mask] += weight_mutations[mutation_mask]
        
        # 3. Randomly remove existing connections (with small probability)
        removal_prob = mutation_rate * 0.2  # Lower probability than mutation
        removal_mask = np.random.random(weights.shape) < removal_prob
        
        # Only consider existing connections for removal
        removal_mask = np.logical_and(removal_mask, existing_connections)
        
        # Don't remove critical connections
        # Ensure each non-input neuron keeps at least one incoming connection
        for i in range(self.snn.num_neurons):
            if i not in self.snn.input_indices:
                # If this would remove all incoming connections, preserve one randomly
                incoming_connections = weights[:, i] != 0
                would_remove = removal_mask[:, i]
                if np.all(np.logical_or(would_remove, ~incoming_connections)):
                    # Keep one random existing connection
                    existing_idx = np.where(incoming_connections)[0]
                    if len(existing_idx) > 0:
                        keep_idx = np.random.choice(existing_idx)
                        removal_mask[keep_idx, i] = False
        
        # Apply removals
        weights[removal_mask] = 0
        
        # 4. Randomly add new connections (with small probability)
        new_connection_prob = mutation_rate * 0.1  # Much lower probability
        potential_new_connections = np.logical_and(weights == 0, np.random.random(weights.shape) < new_connection_prob)
        
        # Don't create connections to input neurons
        input_indices = self.snn.input_indices
        potential_new_connections[:, input_indices] = False
        
        # Initialize new connections with small random weights
        if np.any(potential_new_connections):
            weights[potential_new_connections] = np.random.normal(0, 0.05, size=np.sum(potential_new_connections))
        
        # 5. Threshold mutation (with small probability)
        threshold_mutation_prob = mutation_rate * 0.2
        threshold_mask = np.random.random(self.snn.threshold.shape) < threshold_mutation_prob
        
        # Only mutate non-input neurons
        threshold_mask[self.snn.input_indices] = False
        
        # Generate random mutations for thresholds
        threshold_mutations = np.random.normal(0, 0.05, size=self.snn.threshold.shape)
        
        # Apply threshold mutations
        self.snn.threshold[threshold_mask] += threshold_mutations[threshold_mask]
        
        # Ensure thresholds remain positive
        self.snn.threshold = np.maximum(0.1, self.snn.threshold)
    
    def clone(self) -> 'SnakeAgent':
        """
        Create a deep copy of this agent including its neural network structure.
        
        Returns:
            A new agent with the same neural network weights and parameters
        """
        # Create a new agent with the same hyperparameters
        new_agent = SnakeAgent(
            num_hidden_neurons=self.snn.num_hidden_neurons,
            max_hidden_neurons=self.snn.max_hidden_neurons,
            latency_window=self.latency_window,
            connection_probability=0.4,  # Fixed value
            plasticity_threshold=self.snn.plasticity_threshold,
            neuron_growth_rate=self.snn.neuron_growth_rate,
            connection_growth_rate=self.snn.connection_growth_rate
        )
        
        # Copy the actual neural network weights and connections
        new_agent.snn.weights = self.snn.weights.copy()
        new_agent.snn.threshold = self.snn.threshold.copy()
        new_agent.snn.membrane_potential = self.snn.membrane_potential.copy()
        new_agent.snn.neuron_activity = self.snn.neuron_activity.copy()
        
        # Copy current direction state
        new_agent.current_direction = self.current_direction
        
        return new_agent
    
    @staticmethod
    def crossover(parent1: 'SnakeAgent', parent2: 'SnakeAgent') -> 'SnakeAgent':
        """
        Create a new agent by crossing over two parent agents.
        
        Args:
            parent1: First parent agent
            parent2: Second parent agent
            
        Returns:
            A new agent with mixed properties from both parents
        """
        # Create a new agent with averaged hyperparameters
        child = SnakeAgent(
            num_hidden_neurons=min(parent1.snn.num_hidden_neurons, parent2.snn.num_hidden_neurons),
            max_hidden_neurons=int(max(parent1.snn.max_hidden_neurons, parent2.snn.max_hidden_neurons)),
            latency_window=int((parent1.latency_window + parent2.latency_window) / 2),
            connection_probability=0.4,  # Fixed value
            plasticity_threshold=(parent1.snn.plasticity_threshold + parent2.snn.plasticity_threshold) / 2,
            neuron_growth_rate=(parent1.snn.neuron_growth_rate + parent2.snn.neuron_growth_rate) / 2,
            connection_growth_rate=(parent1.snn.connection_growth_rate + parent2.snn.connection_growth_rate) / 2
        )
        
        # Create connection mask - determines which connections come from which parent
        mask = np.random.random(parent1.snn.weights.shape) > 0.5
        
        # Apply connection crossover using the mask
        child.snn.weights = np.where(mask, parent1.snn.weights, parent2.snn.weights)
        
        # Ensure both parents have same network structure, otherwise 
        # get the shared neurons (up to the minimum number of hidden neurons)
        min_hidden = min(len(parent1.snn.hidden_indices), len(parent2.snn.hidden_indices))
        child.snn.hidden_indices = parent1.snn.hidden_indices[:min_hidden]
        
        # Choose threshold values randomly from either parent
        threshold_mask = np.random.random(parent1.snn.threshold.shape) > 0.5
        child.snn.threshold = np.where(threshold_mask, parent1.snn.threshold, parent2.snn.threshold)
        
        # Inherit current direction from a random parent
        child.current_direction = parent1.current_direction if np.random.random() > 0.5 else parent2.current_direction
        
        return child
    
    @property
    def rng(self) -> np.random.Generator:
        """Get the random number generator from the SNN."""
        return self.snn.rng 