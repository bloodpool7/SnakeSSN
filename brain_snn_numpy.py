import numpy as np
from typing import List, Union, Tuple, Optional

class BrainSNN:
    def __init__(self, num_inputs: int, num_outputs: int, num_hidden_neurons: int,
                 threshold: float = 1.0, 
                 decay_rate: float = 0.2, 
                 connection_probability: float = 0.3,
                 # Plasticity parameters
                 neuron_growth_rate: float = 0.01,
                 connection_growth_rate: float = 0.05,
                 plasticity_threshold: float = 0.7,
                 neuron_activity_decay: float = 0.1,
                 max_hidden_neurons: int = 100):
        """
        Initialize a simple spiking neural network with plasticity.
        
        Args:
            num_inputs: Number of input neurons
            num_outputs: Number of output neurons
            num_hidden_neurons: Number of hidden neurons
            threshold: Activation threshold for neurons
            decay_rate: Rate at which membrane potential decays
            connection_probability: Probability of creating a connection between neurons
            neuron_growth_rate: Probability of generating a new neuron during plasticity
            connection_growth_rate: Probability of forming new connections during plasticity
            plasticity_threshold: Activity threshold that may trigger plasticity
            neuron_activity_decay: Decay factor for neuron activity tracking
            max_hidden_neurons: Maximum number of hidden neurons allowed to grow
        """
        # Network structure
        self.num_neurons = int(num_inputs + num_hidden_neurons + num_outputs)
        self.num_inputs = int(num_inputs)
        self.num_outputs = int(num_outputs)
        self.num_hidden_neurons = int(num_hidden_neurons)
        self.max_hidden_neurons = int(max_hidden_neurons)
        
        # Define indices
        self.input_indices = np.arange(num_inputs)
        self.hidden_indices = np.arange(num_inputs, num_inputs + num_hidden_neurons)
        self.output_indices = np.arange(num_inputs + num_hidden_neurons, self.num_neurons)
        
        # Pre-compute masks for efficient indexing
        self.input_mask = np.zeros(self.num_neurons, dtype=bool)
        self.input_mask[self.input_indices] = True
        self.non_input_mask = ~self.input_mask
        
        # Neuron parameters - using single data type for better memory efficiency
        self.membrane_potential = np.zeros(self.num_neurons, dtype=np.float32)
        self.threshold = np.full(self.num_neurons, threshold, dtype=np.float32)
        self.decay_rate = np.full(self.num_neurons, decay_rate, dtype=np.float32)
        self.has_spiked = np.zeros(self.num_neurons, dtype=bool)
        
        # Pre-allocated arrays for forward pass
        self._decay_factors = 1.0 - self.decay_rate
        
        # Initialize weights using efficient method
        self._initialize_connections(connection_probability)
        
        # Plasticity parameters (evolvable)
        self.neuron_growth_rate = neuron_growth_rate
        self.connection_growth_rate = connection_growth_rate
        self.plasticity_threshold = plasticity_threshold
        
        # Activity tracking for plasticity
        self.neuron_activity = np.zeros(self.num_neurons, dtype=np.float32)
        self.activity_decay = 1.0 - neuron_activity_decay  # Decay factor for neuron activity tracking
        
        # RNG for plasticity
        self.rng = np.random.default_rng()
    
    def _initialize_connections(self, connection_probability: float):
        """Create random connections between neurons with fully vectorized operations."""
        n = self.num_neurons
        
        # Create masks for connection types - all fully vectorized
        # Hidden-to-hidden connections (excluding self-connections)
        hidden_idx = self.hidden_indices
        h_size = len(hidden_idx)
        
        # Create identity-like matrix for hidden neurons
        hidden_diag = np.zeros((n, n), dtype=bool)
        idx = np.ix_(hidden_idx, hidden_idx)
        hidden_diag[idx] = np.eye(h_size, dtype=bool)
        
        # Create full connections between hidden neurons, then remove self-connections
        hidden_to_hidden = np.zeros((n, n), dtype=bool)
        hidden_to_hidden[np.ix_(hidden_idx, hidden_idx)] = True
        hidden_to_hidden &= ~hidden_diag
        
        # Input-to-hidden connections
        input_to_hidden = np.zeros((n, n), dtype=bool)
        input_to_hidden[np.ix_(self.input_indices, hidden_idx)] = True
        
        # Hidden-to-output connections
        hidden_to_output = np.zeros((n, n), dtype=bool)
        hidden_to_output[np.ix_(hidden_idx, self.output_indices)] = True
        
        # Combined allowable connection mask - NO hidden-to-input connections
        allowed_connections = hidden_to_hidden | input_to_hidden | hidden_to_output
        
        # Initialize weights matrix
        self.weights = np.zeros((n, n), dtype=np.float32)
        
        # First pass: Create random connections based on probability
        rng = np.random.default_rng()
        connection_mask = rng.random(size=(n, n)) < connection_probability
        valid_connections = connection_mask & allowed_connections
        self.weights[valid_connections] = rng.uniform(-1.0, 1.0, size=np.sum(valid_connections))
        
        # Second pass: Ensure each non-input neuron has at least one incoming connection
        for i in range(n):
            if not np.any(self.weights[:, i]) and i not in self.input_indices:
                # Find possible sources for this neuron
                if i in self.hidden_indices:
                    # Hidden neurons can receive from inputs and other hidden neurons
                    possible_sources = np.concatenate([
                        self.input_indices,
                        self.hidden_indices[self.hidden_indices != i]
                    ])
                else:  # Output neuron
                    # Output neurons can receive from hidden neurons
                    possible_sources = self.hidden_indices
                
                if len(possible_sources) > 0:
                    # Choose a random source and create a connection
                    source = rng.choice(possible_sources)
                    self.weights[source, i] = rng.uniform(0.5, 1.0)
        
        # Third pass: Ensure network is connected with no isolated subgraphs
        
        # Ensure each input connects to at least one hidden
        for input_idx in self.input_indices:
            if not np.any(self.weights[input_idx, :]):
                # Connect to a random hidden neuron
                target = rng.choice(self.hidden_indices)
                self.weights[input_idx, target] = rng.uniform(0.5, 1.0)
        
        # Ensure each hidden neuron is connected to at least one other neuron
        # (either another hidden neuron, an input, or an output)
        for hidden_idx in self.hidden_indices:
            if not np.any(self.weights[hidden_idx, :]) and not np.any(self.weights[:, hidden_idx]):
                # This neuron is isolated - connect it to another random hidden neuron
                if len(self.hidden_indices) > 1:
                    possible_targets = self.hidden_indices[self.hidden_indices != hidden_idx]
                    target = rng.choice(possible_targets)
                    self.weights[hidden_idx, target] = rng.uniform(0.5, 1.0)
        
        # Ensure each output receives from at least one hidden
        for output_idx in self.output_indices:
            if not np.any(self.weights[:, output_idx]):
                # Connect from a random hidden neuron
                source = rng.choice(self.hidden_indices)
                self.weights[source, output_idx] = rng.uniform(0.5, 1.0)
    
    def forward(self, input_spikes: Union[List[bool], np.ndarray]) -> List[bool]:
        """
        Process one time step and return output spikes.
        
        Args:
            input_spikes: Array-like of boolean values indicating which input neurons are spiking
            
        Returns:
            List of boolean values indicating which output neurons are spiking
        """
        if len(input_spikes) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} input spikes, got {len(input_spikes)}")
        
        # Store previous spikes for propagation
        prev_spikes = self.has_spiked.copy()
        
        # Reset spike status
        self.has_spiked.fill(False)
        
        # Set input neuron spikes (converting input to numpy array if needed)
        if not isinstance(input_spikes, np.ndarray):
            input_spikes = np.array(input_spikes, dtype=bool)
        self.has_spiked[self.input_indices] = input_spikes
        
        # Decay membrane potential (using pre-computed factors)
        self.membrane_potential[self.non_input_mask] *= self._decay_factors[self.non_input_mask]
        
        # Calculate incoming current from PREVIOUS timesteep's spiking neurons
        incoming_current = np.dot(prev_spikes, self.weights)
        self.membrane_potential += incoming_current
        
        # Check for spikes (vectorized)
        spiking_neurons = (self.membrane_potential >= self.threshold) & self.non_input_mask
        
        # Update spike status and reset membrane potential
        self.has_spiked[spiking_neurons] = True
        self.membrane_potential[spiking_neurons] = 0
        
        # Update neuron activity for plasticity
        self.neuron_activity *= self.activity_decay
        self.neuron_activity[self.has_spiked] += 1.0
        
        # Return output spikes
        return self.has_spiked[self.output_indices].tolist()

    def apply_plasticity(self) -> Tuple[int, int]:
        """
        Apply plasticity rules to grow the network.
        
        Returns:
            Tuple of (number of new neurons added, number of new connections added)
        """
        new_neurons = 0
        new_connections = 0
        
        # First, try to grow new neurons
        if self.num_hidden_neurons < self.max_hidden_neurons:
            # Check if we should add a new neuron
            rng = self.rng.random()
            if rng < self.neuron_growth_rate: # self.neuron_growth_rate
                new_neurons = self._grow_neurons()
        
        # Then, try to grow new connections regardless of neuron growth
        new_connections = self._grow_connections()
        
        return new_neurons, new_connections
    
    def _grow_neurons(self) -> int:
        """
        Potentially grow new hidden neurons based on network activity and growth parameters.
        
        Returns:
            Number of new neurons added
        """
        # Don't grow if we're at the maximum
        if self.num_hidden_neurons >= self.max_hidden_neurons:
            return 0
            
        # Determine if growth should happen based on activity and growth rate
        network_activity = np.mean(self.neuron_activity[self.hidden_indices])
        growth_chance = self.neuron_growth_rate * max(0, network_activity - self.plasticity_threshold)

        # Decide how many neurons to add
        num_new_neurons = 0
        if self.rng.random() < 1: # growth_chance
            # Add 1-3 neurons (could be evolved)
            num_new_neurons = self.rng.integers(1, 4)
            num_new_neurons = min(num_new_neurons, int(self.max_hidden_neurons - self.num_hidden_neurons))
            
            if num_new_neurons > 0:
                self._add_neurons(num_new_neurons)
                
        return num_new_neurons
    
    def _add_neurons(self, num_new_neurons: int):
        """
        Add new neurons to the network while properly preserving output connections.
        
        Args:
            num_new_neurons: Number of neurons to add
        """
        if num_new_neurons <= 0:
            return
        
        old_num_neurons = self.num_neurons
        old_num_hidden = self.num_hidden_neurons
        
        # Store old output indices before we change them
        old_output_indices = self.output_indices.copy()
        
        # Update neuron counts
        self.num_hidden_neurons = int(self.num_hidden_neurons + num_new_neurons)
        self.num_neurons = int(self.num_inputs + self.num_hidden_neurons + self.num_outputs)
        
        # Calculate new indices
        self.hidden_indices = np.arange(self.num_inputs, self.num_inputs + self.num_hidden_neurons)
        self.output_indices = np.arange(self.num_inputs + self.num_hidden_neurons, self.num_neurons)
        
        # Create masks for different neuron types
        self.input_mask = np.zeros(self.num_neurons, dtype=bool)
        self.input_mask[self.input_indices] = True
        
        self.output_mask = np.zeros(self.num_neurons, dtype=bool)
        self.output_mask[self.output_indices] = True
        
        self.hidden_mask = np.zeros(self.num_neurons, dtype=bool)
        self.hidden_mask[self.hidden_indices] = True
        
        self.non_input_mask = ~self.input_mask
        
        # Create new arrays with expanded size
        new_weights = np.zeros((self.num_neurons, self.num_neurons))
        new_membrane_potential = np.zeros(self.num_neurons)
        new_threshold = np.zeros(self.num_neurons)
        new_decay_rate = np.zeros(self.num_neurons)
        new_has_spiked = np.zeros(self.num_neurons, dtype=bool)
        new_neuron_activity = np.zeros(self.num_neurons)
        
        # Copy existing neuron properties
        new_membrane_potential[:old_num_neurons] = self.membrane_potential
        new_threshold[:old_num_neurons] = self.threshold
        new_decay_rate[:old_num_neurons] = self.decay_rate
        new_has_spiked[:old_num_neurons] = self.has_spiked
        new_neuron_activity[:old_num_neurons] = self.neuron_activity
        
        # Copy existing weights for all neurons except outputs
        for i in range(old_num_neurons):
            if i not in old_output_indices:  # For non-output neurons
                for j in range(old_num_neurons):
                    new_weights[i, j] = self.weights[i, j]
        
        # Special handling for output neurons: their indices have changed
        # Map old output indices to new output indices
        for i, old_idx in enumerate(old_output_indices):
            new_idx = self.output_indices[i]
            
            # Copy incoming connections TO this output neuron
            for j in range(old_num_neurons):
                if j != old_idx:  # Skip self-connections
                    new_weights[j, new_idx] = self.weights[j, old_idx]
            
            # Copy outgoing connections FROM this output neuron
            for j in range(old_num_neurons):
                if j != old_idx:  # Skip self-connections
                    new_weights[new_idx, j] = self.weights[old_idx, j]
        
        # Initialize new neurons (vectorized)
        new_idx_start = self.num_inputs + old_num_hidden
        new_idx_end = self.num_inputs + self.num_hidden_neurons
        new_neuron_indices = np.arange(new_idx_start, new_idx_end)
        
        # Initialize properties for new neurons with some randomness
        new_threshold[new_neuron_indices] = self.threshold[self.hidden_indices[0]] + self.rng.normal(0, 0.1, num_new_neurons)
        new_decay_rate[new_neuron_indices] = self.decay_rate[self.hidden_indices[0]] + self.rng.normal(0, 0.05, num_new_neurons)
        
        # Create connections for new neurons (vectorized)
        for i, new_idx in enumerate(new_neuron_indices):
            # Connect from some existing hidden neurons to this new neuron
            if old_num_hidden > 0:
                connect_from = self.rng.choice(
                    self.hidden_indices[:old_num_hidden], 
                    size=max(1, int(old_num_hidden * 0.3)),
                    replace=False
                )
                new_weights[connect_from, new_idx] = self.rng.normal(0, 0.5, len(connect_from))
            
            # Connect from this new neuron to some existing hidden neurons
            if old_num_hidden > 0:
                connect_to_hidden = self.rng.choice(
                    self.hidden_indices[:old_num_hidden], 
                    size=max(1, int(old_num_hidden * 0.3)),
                    replace=False
                )
                new_weights[new_idx, connect_to_hidden] = self.rng.normal(0, 0.5, len(connect_to_hidden))
        
        # Update arrays
        self.weights = new_weights
        self.membrane_potential = new_membrane_potential
        self.threshold = new_threshold
        self.decay_rate = new_decay_rate
        self.has_spiked = new_has_spiked
        self.neuron_activity = new_neuron_activity
        self._decay_factors = 1.0 - self.decay_rate  # Update decay factors
    
    def _grow_connections(self) -> int:
        """
        Grow new connections between hidden neurons based on their activity levels.
        
        Returns:
            Number of new connections added
        """
        # 1. Check if we have enough hidden neurons
        if len(self.hidden_indices) < 2:
            return 0
            
        # 2. Find active hidden neurons
        active_hidden = self.hidden_indices[self.neuron_activity[self.hidden_indices] > self.plasticity_threshold]
        
        # If not enough active hidden neurons, skip
        if len(active_hidden) < 2:
            return 0
            
        # 3. Determine connection growth probability based on hidden neuron activity
        hidden_activity = np.mean(self.neuron_activity[self.hidden_indices])
        
        # Calculate growth probability based on activity
        growth_chance = self.connection_growth_rate * max(0, hidden_activity - self.plasticity_threshold)
        
        # Apply a basic chance to create connections regardless of activity
        base_chance = 0.05 * self.connection_growth_rate
        
        # Combined chance (ensure it's between 0 and 1)
        connection_probability = min(1.0, max(0.0, growth_chance + base_chance))
        
        # Skip if the random chance is too low
        if self.rng.random() > connection_probability:
            return 0
            
        # 4. Create new connections between hidden neurons only
        num_connections_added = 0
        
        # Create an "allowed connections" mask for hidden-to-hidden only
        n = self.num_neurons
        allowed_connections = np.zeros((n, n), dtype=bool)
        
        # Hidden neurons can connect to hidden neurons (except self)
        hidden_to_hidden = np.zeros((n, n), dtype=bool)
        hidden_to_hidden[np.ix_(self.hidden_indices, self.hidden_indices)] = True
        hidden_to_hidden[np.diag_indices(n)] = False  # No self-connections
        allowed_connections = hidden_to_hidden  # Only hidden-to-hidden allowed
        
        # Find existing connections to avoid duplicating
        existing_connections = self.weights != 0
        
        # Identify potential new connections
        potential_connections = allowed_connections & ~existing_connections
        
        # If no potential connections, return
        if not np.any(potential_connections):
            return 0
            
        # Determine how many connections to add (more active network = more connections)
        # Scale by the network activity
        max_new_connections = max(3, int(10 * hidden_activity))
        
        # Find all potential connection pairs
        potential_pairs = np.where(potential_connections)
        
        # If no potential connections, return
        if len(potential_pairs[0]) == 0:
            return 0

        # Randomly select connection pairs to add
        num_to_add = min(max_new_connections, len(potential_pairs[0]))
        if num_to_add == 0:
            return 0
        
        # Select random connection indices
        connection_indices = self.rng.choice(
            len(potential_pairs[0]),
            size=num_to_add,
            replace=False
        )
        
        # Add the new connections
        for idx in connection_indices:
            source = potential_pairs[0][idx]
            target = potential_pairs[1][idx]
            
            # Use activity-based weights - stronger connections between highly active neurons
            source_activity = self.neuron_activity[source]
            target_activity = self.neuron_activity[target]
            
            # Calculate weight based on activity correlation
            # More active neurons get stronger connections
            base_weight = self.rng.uniform(0.3, 1.0)
            activity_factor = min(1.0, (source_activity + target_activity) / (2 * self.plasticity_threshold + 1e-6))
            weight = base_weight * (0.5 + 0.5 * activity_factor)
            
            # Hidden to hidden has both excitatory and inhibitory connections
            if self.rng.random() < 0.5:
                weight = abs(weight)
            else:
                weight = -abs(weight)
            
            # Apply the new connection
            self.weights[source, target] = weight
            num_connections_added += 1
            
            # Add reciprocal connection with 50% probability
            if self.rng.random() < 0.5:
                reciprocal_weight = weight * self.rng.uniform(0.7, 1.3)  # Slightly different weight
                self.weights[target, source] = reciprocal_weight
                num_connections_added += 1
        
        return num_connections_added

    def reset(self):
        """Reset the network state."""
        self.membrane_potential.fill(0.0)
        self.has_spiked.fill(False)
        self.neuron_activity.fill(0.0)
        
    def get_plasticity_params(self) -> dict:
        """
        Get the current plasticity parameters as a dictionary.
        This is useful for evolutionary algorithms to mutate these parameters.
        
        Returns:
            Dictionary of plasticity parameters
        """
        return {
            'neuron_growth_rate': self.neuron_growth_rate,
            'connection_growth_rate': self.connection_growth_rate,
            'plasticity_threshold': self.plasticity_threshold,
            'max_hidden_neurons': self.max_hidden_neurons,
            'neuron_activity_decay': 1.0 - self.activity_decay
        }
        
    def set_plasticity_params(self, params: dict):
        """
        Set plasticity parameters from a dictionary.
        This is useful for evolutionary algorithms to update parameters.
        
        Args:
            params: Dictionary of plasticity parameters to set
        """
        if 'neuron_growth_rate' in params:
            self.neuron_growth_rate = params['neuron_growth_rate']
        if 'connection_growth_rate' in params:
            self.connection_growth_rate = params['connection_growth_rate']
        if 'plasticity_threshold' in params:
            self.plasticity_threshold = params['plasticity_threshold']
        if 'max_hidden_neurons' in params:
            self.max_hidden_neurons = params['max_hidden_neurons']
        if 'neuron_activity_decay' in params:
            self.activity_decay = 1.0 - params['neuron_activity_decay']
