import numpy as np
import time
import pickle
import argparse
from snake_environment import SnakeEnvironment, Direction
from snake_agent import SnakeAgent

def print_separator(title=None):
    """Print a separator line with optional title."""
    width = 80
    if title:
        padding = (width - len(title) - 2) // 2
        print("\n" + "=" * padding + f" {title} " + "=" * padding)
    else:
        print("\n" + "=" * width)

def display_state(state, verbose=False):
    """Display the state information in a readable format."""
    print_separator("STATE")
    
    # Display direction
    direction_names = ["UP", "RIGHT", "DOWN", "LEFT"]
    direction_idx = np.argmax(state["direction"])
    direction_str = direction_names[direction_idx]
    print(f"Direction: {direction_str} {state['direction']}")
    
    # Display danger
    danger = state["danger"]
    print(f"Danger: [Forward: {danger[0]}, Right: {danger[1]}, Left: {danger[2]}]")
    
    # Display food direction
    food_dir = state["food_direction"]
    print(f"Food Direction: [{food_dir[0]:.2f}, {food_dir[1]:.2f}]")
    
    # Display vision rays if verbose
    if verbose:
        vision = state["vision"].reshape(-1, 3)
        print("\nVision Rays:")
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        print("         | Wall Dist | Food Dist | Body Dist |")
        for i, ray in enumerate(vision):
            print(f"{directions[i]:>8} | {ray[0]:9.2f} | {ray[1]:9.2f} | {ray[2]:9.2f} |")

def display_snn_state(agent):
    """Display the current state of the agent's SNN."""
    print_separator("NEURAL ACTIVITY")
    
    # Get SNN information
    snn = agent.snn
    
    # Show output neuron membrane potentials
    output_potentials = snn.membrane_potential[snn.output_indices]
    output_names = ["UP", "RIGHT", "DOWN", "LEFT"]
    print("Output Neuron Potentials:")
    for i, (name, potential) in enumerate(zip(output_names, output_potentials)):
        threshold = snn.threshold[snn.output_indices[i]]
        percentage = (potential / threshold) * 100 if threshold > 0 else 0
        bar = "#" * int(percentage / 5)
        print(f"{name:>5}: {potential:.3f}/{threshold:.3f} [{bar:<20}] {percentage:.1f}%")
    
    # Show active hidden neurons count
    active_hidden = np.sum(snn.neuron_activity[snn.hidden_indices] > 0)
    total_hidden = len(snn.hidden_indices)
    print(f"\nHidden Neurons: {active_hidden}/{total_hidden} active")
    
    # Show neuron and connection counts
    total_connections = np.sum(snn.weights != 0)
    hidden_connections = np.sum(snn.weights[np.ix_(snn.hidden_indices, snn.hidden_indices)] != 0)
    print(f"Total Connections: {total_connections} (Hidden-to-Hidden: {hidden_connections})")

def display_action_info(action, reward, latency_info=None):
    """Display information about the action taken and reward received."""
    print_separator("ACTION & REWARD")
    
    action_names = ["UP", "RIGHT", "DOWN", "LEFT"]
    print(f"Action Chosen: {action_names[action]} ({action})")
    
    if latency_info is not None:
        print("\nLatency Information:")
        for i, (name, time) in enumerate(zip(action_names, latency_info)):
            if time <= 20:  # Only show if spike occurred within window
                print(f"{name:>5}: First spike at t={time}")
            else:
                print(f"{name:>5}: No spike")
    
    # Display reward with color and explanation
    reward_explanation = ""
    if reward == 1.0:
        reward_explanation = "(Food eaten!)"
    elif reward == 0.03:
        reward_explanation = "(Moving toward food)"
    elif reward == 0.01:
        reward_explanation = "(Survival)"
    elif reward == -0.01:
        reward_explanation = "(Moving away from food)"
    elif reward == -1.0:
        reward_explanation = "(Game over!)"
    
    print(f"\nReward: {reward:.2f} {reward_explanation}")

def slow_run(agent_path=None, speed=1.0, max_steps=1000, verbose=False):
    """
    Run a detailed step-by-step visualization of an agent.
    
    Args:
        agent_path: Path to a saved agent, or None to create a new one
        speed: Delay between steps in seconds (higher = slower)
        max_steps: Maximum steps to run
        verbose: Whether to show detailed vision information
    """
    # Load or create agent
    if agent_path:
        print(f"Loading agent from {agent_path}...")
        with open(agent_path, "rb") as f:
            agent = pickle.load(f)
    else:
        print("Creating a new agent...")
        agent = SnakeAgent(
            num_hidden_neurons=50,
            max_hidden_neurons=200,
            latency_window=20,
            connection_probability=0.4,
            plasticity_threshold=0.5
        )
    
    # Create environment
    env = SnakeEnvironment(width=15, height=15)
    state = env.reset()
    
    # Game statistics
    total_reward = 0
    steps = 0
    
    # Introduction and controls
    print("\033[H\033[J", end="")  # Clear terminal initially
    print_separator("CONTROLS")
    print("ENTER: Next step")
    print("c: Clear screen")
    print("v: Toggle verbose mode")
    print("s: Show game state only")
    print("n: Show neural network state only")
    print("a: Show action & reward info only")
    print("q: Quit")
    print("\nPress ENTER to start...")
    input()
    
    # Main game loop
    done = False
    current_verbose = verbose
    show_all = True
    while not done and steps < max_steps:
        # Clear screen only when explicitly requested
        print("\033[H\033[J", end="")  # Clear terminal
        
        # Always show the game state
        env.render(mode="terminal")
        
        # Display game information
        print(f"Step: {steps} | Score: {env.score} | Total Reward: {total_reward:.2f}")
        
        # Get input to determine what to display and when to continue
        input_prompt = "\nENTER: next step, c: clear, v: verbose, s: state, n: neural, a: action, q: quit: "
        user_input = input(input_prompt)
        
        # Process user input
        if user_input.lower() == 'q':
            break
        elif user_input.lower() == 'c':
            print("\033[H\033[J", end="")  # Clear terminal
            continue
        elif user_input.lower() == 'v':
            current_verbose = not current_verbose
            print(f"Verbose mode {'enabled' if current_verbose else 'disabled'}")
            continue
        elif user_input.lower() == 's':
            # Show only game state
            show_all = False
            display_state(state, verbose=current_verbose)
            input("Press ENTER to continue...")
            continue
        elif user_input.lower() == 'n':
            # Show only neural network state
            show_all = False
            
            # Process state (need to do this to see neural state)
            input_spikes = agent.process_state(state)
            if agent.using_latency_coding:
                agent.snn.reset()
                # Run for a few steps to update membrane potentials
                for t in range(3):
                    agent.snn.forward(input_spikes)
            
            display_snn_state(agent)
            input("Press ENTER to continue...")
            continue
        elif user_input.lower() == 'a':
            # Show only action & reward (need to compute these first)
            show_all = False
            action = agent.act(state)
            next_state, reward, _, _ = env.step(action)
            env.snake_body.pop(0)  # Undo the step
            display_action_info(action, reward)
            input("Press ENTER to continue...")
            continue
        
        # If we get here, we're doing a full step with all information
        show_all = True
        
        # Display state information
        if show_all:
            display_state(state, verbose=current_verbose)
            input("Press ENTER to continue to decision process...")
        
        # Process agent's decision (with latency tracking)
        if show_all:
            print_separator("DECISION PROCESS")
        
        # Get input spikes
        input_spikes = agent.process_state(state)
        
        if agent.using_latency_coding:
            # Reset SNN
            agent.snn.reset()
            
            # Initialize latency tracking
            first_spike_time = np.ones(4) * (agent.latency_window + 1)
            
            # Run SNN for multiple time steps
            all_outputs = []
            for t in range(agent.latency_window):
                outputs = agent.snn.forward(input_spikes)
                all_outputs.append(outputs.copy())
                
                # Record first spike times
                new_spikes = np.logical_and(outputs, first_spike_time > agent.latency_window)
                first_spike_mask = np.where(new_spikes)[0]
                
                if len(first_spike_mask) > 0:
                    for idx in first_spike_mask:
                        first_spike_time[idx] = t
                
                # Show network state at this time step
                if show_all:
                    print(f"\nTime Step {t}:")
                    print(f"Outputs: {outputs}")
                    
                    # If any output has spiked, show it
                    if np.any(outputs):
                        action_names = ["UP", "RIGHT", "DOWN", "LEFT"]
                        spike_indices = np.where(outputs)[0]
                        spike_names = [action_names[i] for i in spike_indices]
                        print(f"Spike detected: {', '.join(spike_names)}")
                
                # Break early if we have at least one spike
                if np.any(first_spike_time <= agent.latency_window) and t > 5:  # Run at least 5 steps
                    if show_all:
                        print(f"First spike detected at t={np.min(first_spike_time[first_spike_time <= agent.latency_window])}")
                    break
            
            # Choose action based on latency
            opposite_dir = (agent.current_direction + 2) % 4
            first_spike_time[opposite_dir] = agent.latency_window + 1  # Prevent 180-degree turn
            
            if np.any(first_spike_time <= agent.latency_window):
                action = np.argmin(first_spike_time)
            else:
                action = agent.current_direction
            
            agent.current_direction = action
        else:
            action = agent.act(state)
        
        # Allow user to review decision process before continuing
        if show_all:
            input("Press ENTER to continue to neural activity...")
        
        # Display SNN state
        if show_all:
            display_snn_state(agent)
            input("Press ENTER to continue to next step and reward...")
        
        # Take step in environment
        next_state, reward, done, info = env.step(action)
        
        # Display action and reward information
        if show_all:
            if agent.using_latency_coding:
                display_action_info(action, reward, latency_info=first_spike_time)
            else:
                display_action_info(action, reward)
            input("Press ENTER to complete this step...")
        
        # Update agent
        agent.update(reward)
        
        # Update tracking variables
        state = next_state
        total_reward += reward
        steps += 1
        
        # Add delay based on speed
        time.sleep(1.0 / speed)
    
    # Game over - show final state
    print("\033[H\033[J", end="")  # Clear terminal
    env.render(mode="terminal")
    
    # Display game over information
    print_separator("GAME OVER")
    print(f"Final Score: {env.score}")
    print(f"Steps Taken: {steps}")
    print(f"Total Reward: {total_reward:.2f}")
    
    if done:
        print("Reason: ", end="")
        head_x, head_y = env.snake_body[0]
        if head_x < 0 or head_x >= env.width or head_y < 0 or head_y >= env.height:
            print("Hit wall")
        elif env.snake_body[0] in env.snake_body[1:]:
            print("Hit self")
        elif env.steps_without_food >= env.max_steps_without_food:
            print("Too many steps without food")
        else:
            print("Unknown")
    else:
        print(f"Reason: Reached maximum steps ({max_steps})")
        
    print("\nPress ENTER to exit...")
    input()

def fast_run(agent_path=None, speed=1.0, max_steps=1000):
    """
    Run a simple visualization of an agent without detailed step information.
    Just press Enter to advance to the next step.
    
    Args:
        agent_path: Path to a saved agent, or None to create a new one
        speed: Delay between steps in seconds (higher = slower)
        max_steps: Maximum steps to run
    """
    # Load or create agent
    if agent_path:
        print(f"Loading agent from {agent_path}...")
        with open(agent_path, "rb") as f:
            agent = pickle.load(f)
    else:
        print("Creating a new agent...")
        agent = SnakeAgent(
            num_hidden_neurons=50,
            max_hidden_neurons=200,
            latency_window=20,
            connection_probability=0.4,
            plasticity_threshold=0.5
        )
    
    # Create environment
    env = SnakeEnvironment(width=15, height=15)
    state = env.reset()
    
    # Game statistics
    total_reward = 0
    steps = 0
    
    # Introduction and controls
    print("\033[H\033[J", end="")  # Clear terminal
    print_separator("FAST MODE CONTROLS")
    print("ENTER: Run next step")
    print("a: Auto-run (no pausing)")
    print("p: Pause auto-run")
    print("q: Quit")
    print("\nPress ENTER to start...")
    input()
    
    # Main game loop
    done = False
    auto_run = False
    while not done and steps < max_steps:
        # Clear screen
        print("\033[H\033[J", end="")  # Clear terminal
        
        # Show the game state
        env.render(mode="terminal")
        
        # Display basic game information
        print(f"Step: {steps} | Score: {env.score} | Total Reward: {total_reward:.2f}")
        
        # Make agent decision
        action = agent.act(state)
        
        # Show action direction
        action_names = ["UP", "RIGHT", "DOWN", "LEFT"]
        print(f"Action: {action_names[action]}")
        
        # Take step in environment
        next_state, reward, done, info = env.step(action)
        
        # Update agent
        agent.update(reward)
        
        # Update tracking variables
        state = next_state
        total_reward += reward
        steps += 1
        
        # Get user input (if not in auto mode)
        if not auto_run:
            user_input = input("\nENTER: next step, a: auto-run, q: quit: ")
            if user_input.lower() == 'q':
                break
            elif user_input.lower() == 'a':
                auto_run = True
                print("Auto-run enabled. Press 'p' to pause.")
        else:
            # In auto mode, check for key press without waiting
            import sys
            import select
            
            # Check if there's input available (non-blocking)
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key == 'p':
                    auto_run = False
                    print("Auto-run paused. Press ENTER to continue or 'a' to resume auto-run.")
                    input()
            
            # Add delay in auto mode
            time.sleep(0.5 / speed)
    
    # Game over - show final state
    print("\033[H\033[J", end="")  # Clear terminal
    env.render(mode="terminal")
    
    # Display game over information
    print_separator("GAME OVER")
    print(f"Final Score: {env.score}")
    print(f"Steps Taken: {steps}")
    print(f"Total Reward: {total_reward:.2f}")
    
    if done:
        print("Reason: ", end="")
        head_x, head_y = env.snake_body[0]
        if head_x < 0 or head_x >= env.width or head_y < 0 or head_y >= env.height:
            print("Hit wall")
        elif env.snake_body[0] in env.snake_body[1:]:
            print("Hit self")
        elif env.steps_without_food >= env.max_steps_without_food:
            print("Too many steps without food")
        else:
            print("Unknown")
    else:
        print(f"Reason: Reached maximum steps ({max_steps})")
        
    print("\nPress ENTER to exit...")
    input()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor an agent playing Snake with detailed information")
    parser.add_argument("--agent", type=str, default=None, help="Path to saved agent file (omit to create new)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed multiplier (higher = faster)")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum number of steps to run")
    parser.add_argument("--verbose", action="store_true", help="Show detailed vision information")
    parser.add_argument("--fast", action="store_true", help="Use fast mode (simplified display)")
    
    args = parser.parse_args()
    
    if args.fast:
        fast_run(
            agent_path=args.agent,
            speed=args.speed,
            max_steps=args.max_steps
        )
    else:
        slow_run(
            agent_path=args.agent,
            speed=args.speed,
            max_steps=args.max_steps,
            verbose=args.verbose
        ) 