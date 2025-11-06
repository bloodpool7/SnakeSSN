import numpy as np
import time
import pickle
import os
import multiprocessing as mp
from typing import List, Tuple, Any
from functools import partial
from snake_environment import SnakeEnvironment
from snake_agent import SnakeAgent
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

# Evaluation function that can be pickled for multiprocessing
def evaluate_agent_mp(agent_and_params: Tuple[SnakeAgent, dict]) -> Tuple[int, float]:
    """
    Evaluate an agent's performance over multiple episodes.
    
    Args:
        agent_and_params: Tuple containing (agent, params_dict)
        
    Returns:
        Tuple of (agent_index, fitness_score)
    """
    agent, params = agent_and_params
    agent_index = params["agent_index"]
    episodes_per_eval = params["episodes_per_eval"]
    max_steps = params["max_steps"]
    grid_size = params["grid_size"]
    render = params["render"]
    render_mode = params.get("render_mode", "terminal")
    
    env = SnakeEnvironment(width=grid_size, height=grid_size)
    
    # Run multiple episodes for more reliable evaluation
    total_score = 0
    total_steps = 0
    total_reward = 0
    total_food_eaten = 0
    longest_episode = 0
    
    for episode in range(episodes_per_eval):
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        food_eaten = 0
        
        while not done and steps < max_steps:
            # Choose action
            action = agent.act(state)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Update agent
            agent.update(reward)
            
            # Render if requested
            if render:
                env.render(mode=render_mode)
                time.sleep(0.05)
            
            # Track food eaten
            if reward == 1.0:  # Eating food gives reward of 1.0
                food_eaten += 1
            
            # Update tracking variables
            state = next_state
            episode_reward += reward
            steps += 1
            
            # If episode is done, record metrics
            if done:
                total_score += info["score"]
                total_steps += steps
                total_reward += episode_reward
                total_food_eaten += food_eaten
                longest_episode = max(longest_episode, steps)
    
    # Calculate fitness components
    avg_score = total_score / episodes_per_eval
    avg_steps = total_steps / episodes_per_eval
    avg_food = total_food_eaten / episodes_per_eval
    
    # New fitness formula that's always positive:
    # 1. Each piece of food is worth 10 points
    # 2. Survival time is worth up to 5 points (normalized by max steps)
    # 3. Longest episode gives bonus points
    # 4. Base fitness of 1 to keep all values positive
    survival_factor = min(5.0, avg_steps / 100)  # Cap at 5 points
    longest_bonus = min(3.0, longest_episode / 200)  # Cap at 3 points
    
    fitness = 1.0 + (avg_food * 10.0) + survival_factor + longest_bonus
    
    return agent_index, fitness

class EvolutionTrainer:
    def __init__(
        self,
        population_size: int = 20,
        num_generations: int = 100,
        episodes_per_eval: int = 10,
        mutation_rate: float = 0.1,
        max_steps: int = 500,
        grid_size: int = 10,
        render_best: bool = True,
        save_best: bool = True,
        save_path: str = "best_agent.pkl",
        use_parallel: bool = True,
        num_processes: int = None,
        render_mode: str = "terminal"
    ):
        """
        Train Snake agents using evolutionary techniques.
        
        Args:
            population_size: Size of the population
            num_generations: Number of generations to train
            episodes_per_eval: Episodes to evaluate each agent on
            mutation_rate: Probability of parameter mutation
            max_steps: Maximum steps per episode
            grid_size: Size of the Snake grid
            render_best: Whether to render the best agent
            save_best: Whether to save the best agent
            save_path: Path to save the best agent
            use_parallel: Whether to use parallel processing for agent evaluation within a generation
            num_processes: Number of processes to use (defaults to CPU count)
            render_mode: How to render the game ('terminal' or 'human')
        """
        self.population_size = population_size
        self.num_generations = num_generations
        self.episodes_per_eval = episodes_per_eval
        self.mutation_rate = mutation_rate
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.render_best = render_best
        self.save_best = save_best
        self.save_path = save_path
        self.use_parallel = use_parallel
        self.render_mode = render_mode
        
        # Track statistics
        self.best_fitness_per_gen = []
        self.avg_fitness_per_gen = []
        
        # For parallel processing
        if num_processes is None:
            self.num_processes = mp.cpu_count() - 1 or 1  # Leave one core free
        else:
            self.num_processes = num_processes
        
        print(f"Using {self.num_processes} processes for parallel evaluation within each generation")
        
        # Create initial population
        self.population = self.create_population(population_size)
        self.best_agent = None
        self.best_fitness = 0
    
    def create_population(self, size: int) -> List[SnakeAgent]:
        """Create a population of agents with varied parameters."""
        population = []
        # Add progress bar for population creation
        for _ in tqdm(range(size), desc="Creating population", leave=False):
            # Create agent with randomized parameters
            agent = SnakeAgent(
                num_hidden_neurons=100,
                max_hidden_neurons=200,
                latency_window=np.random.randint(10, 30),
                connection_probability=np.random.uniform(0.2, 0.6),
                plasticity_threshold=np.random.uniform(0.3, 0.7),
                neuron_growth_rate=np.random.uniform(0.005, 0.05),
                connection_growth_rate=np.random.uniform(0.01, 0.1)
            )
            population.append(agent)
        return population
    
    def evaluate_agent(self, agent: SnakeAgent, render: bool = False) -> float:
        """
        Evaluate a single agent's performance.
        
        Args:
            agent: The agent to evaluate
            render: Whether to render the environment
            
        Returns:
            fitness: Average fitness score
        """
        # Create evaluation parameters
        params = {
            "agent_index": 0,  # Not important for single evaluation
            "episodes_per_eval": self.episodes_per_eval,
            "max_steps": self.max_steps,
            "grid_size": self.grid_size,
            "render": render,
            "render_mode": self.render_mode
        }
        
        # Use the same evaluation function as in parallel
        _, fitness = evaluate_agent_mp((agent, params))
        
        if render:
            # For rendering, we need a separate evaluation to show visuals
            env = SnakeEnvironment(width=self.grid_size, height=self.grid_size)
            state = env.reset()
            done = False
            steps = 0
            score = 0
            
            # Create progress bar for steps when rendering
            pbar = tqdm(total=self.max_steps, desc="Testing agent", leave=False)
            
            while not done and steps < self.max_steps:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                agent.update(reward)
                state = next_state
                steps += 1
                
                # Update score and progress bar
                if "score" in info:
                    score = info["score"]
                    pbar.set_description(f"Testing agent (Score: {score})")
                
                # Update progress bar
                pbar.update(1)
                
                env.render(mode=self.render_mode)
                time.sleep(0.1)
            
            pbar.close()
            tqdm.write(f"Evaluation finished: Score = {score}, Steps = {steps}")
        
        return fitness
    
    def evaluate_population(self) -> List[Tuple[SnakeAgent, float]]:
        """
        Evaluate the entire population, either in parallel or sequentially.
        
        Returns:
            evaluated_pop: List of (agent, fitness) tuples
        """
        if self.use_parallel and self.num_processes > 1:
            return self._evaluate_population_parallel()
        else:
            return self._evaluate_population_sequential()
    
    def _evaluate_population_sequential(self) -> List[Tuple[SnakeAgent, float]]:
        """Evaluate population sequentially (one agent at a time)."""
        evaluated_pop = []
        
        # Use tqdm for progress tracking
        for i, agent in enumerate(tqdm(self.population, desc="Evaluating agents")):
            fitness = self.evaluate_agent(agent)
            evaluated_pop.append((agent, fitness))
            tqdm.write(f"Agent {i+1}/{len(self.population)}: Fitness = {fitness:.2f}")
        
        # Sort by fitness (descending)
        evaluated_pop.sort(key=lambda x: x[1], reverse=True)
        
        return evaluated_pop
    
    def _evaluate_population_parallel(self) -> List[Tuple[SnakeAgent, float]]:
        """Evaluate population in parallel using multiprocessing."""
        # Create parameters for each agent
        agent_params = []
        for i, agent in enumerate(self.population):
            params = {
                "agent_index": i,
                "episodes_per_eval": self.episodes_per_eval,
                "max_steps": self.max_steps,
                "grid_size": self.grid_size,
                "render": False,  # Never render in parallel processes
                "render_mode": self.render_mode
            }
            agent_params.append((agent, params))
        
        # Create a process pool and map evaluation function to all agents
        start_time = time.time()
        tqdm.write(f"Starting parallel evaluation of {len(self.population)} agents...")
        
        with mp.Pool(processes=self.num_processes) as pool:
            results = list(tqdm(
                pool.imap(evaluate_agent_mp, agent_params),
                desc="Evaluating agents in parallel",
                total=len(agent_params)
            ))
        
        # Organize results by agent index
        fitness_results = {}
        for agent_idx, fitness in results:
            fitness_results[agent_idx] = fitness
        
        # Create final evaluated population sorted by fitness
        evaluated_pop = [(self.population[idx], fitness) for idx, fitness in fitness_results.items()]
        evaluated_pop.sort(key=lambda x: x[1], reverse=True)
        
        elapsed = time.time() - start_time
        tqdm.write(f"Parallel evaluation completed in {elapsed:.2f} seconds")
        
        # Print fitness of each agent
        for i, (_, fitness) in enumerate(evaluated_pop):
            tqdm.write(f"Agent {i+1}/{len(self.population)}: Fitness = {fitness:.2f}")
        
        return evaluated_pop
    
    def selection(self, evaluated_pop: List[Tuple[SnakeAgent, float]]) -> List[SnakeAgent]:
        """
        Select agents for the next generation.
        
        Args:
            evaluated_pop: List of (agent, fitness) tuples
            
        Returns:
            selected: List of selected agents (only top 2)
        """
        # Select only the top 2 performers
        return [agent for agent, _ in evaluated_pop[:2]]
    
    def crossover_and_mutation(self, selected: List[SnakeAgent]) -> List[SnakeAgent]:
        """
        Create a new population with:
        1. Half derived from top performers (with mutations)
        2. Half completely new random agents
        
        This hybrid approach helps escape local minima by maintaining exploration.
        
        Args:
            selected: List of selected agents (top 2)
            
        Returns:
            new_population: New population with both evolved and fresh agents
        """
        # Ensure we have at least 1 parent
        if not selected:
            raise ValueError("No selected agents provided")
        
        # Use only the best parent (first in the list)
        best_parent = selected[0]
        
        # Create new population starting with exact clones of the selected parents
        new_population = [best_parent.clone()]
        
        # If we have at least 2 selected agents, include the second best too
        if len(selected) >= 2:
            new_population.append(selected[1].clone())
        
        # Calculate how many evolved agents vs. new random agents to create
        total_remaining = self.population_size - len(new_population)
        evolved_count = total_remaining // 2
        random_count = total_remaining - evolved_count
        
        # Fill half with evolved agents (clones and mutations of the best parent)
        for _ in tqdm(range(evolved_count), desc="Creating evolved agents", leave=False):
            # Clone the best parent
            child = best_parent.clone()
            
            # Apply mutation to create diversity
            child.mutate_parameters(self.mutation_rate)
            
            # Add to new population
            new_population.append(child)
        
        # Fill the other half with completely new random agents
        for _ in tqdm(range(random_count), desc="Creating random agents", leave=False):
            # Create fresh agent with randomized parameters
            new_agent = SnakeAgent(
                num_hidden_neurons=100,
                max_hidden_neurons=200,
                latency_window=np.random.randint(10, 30),
                connection_probability=np.random.uniform(0.2, 0.6),
                plasticity_threshold=np.random.uniform(0.3, 0.7),
                neuron_growth_rate=np.random.uniform(0.005, 0.05),
                connection_growth_rate=np.random.uniform(0.01, 0.1)
            )
            new_population.append(new_agent)
            
        return new_population
    
    def train(self):
        """Run the evolutionary training process."""
        # Use trange instead of range for a progress bar on generations
        for generation in trange(self.num_generations, desc="Training generations"):
            tqdm.write(f"\nGeneration {generation+1}/{self.num_generations}")
            
            # Evaluate population
            evaluated_pop = self.evaluate_population()
            
            # Update statistics
            fitness_values = [fitness for _, fitness in evaluated_pop]
            best_fitness = fitness_values[0]
            avg_fitness = np.mean(fitness_values)
            
            self.best_fitness_per_gen.append(best_fitness)
            self.avg_fitness_per_gen.append(avg_fitness)
            
            tqdm.write(f"Best fitness: {best_fitness:.2f}, Avg fitness: {avg_fitness:.2f}")
            tqdm.write(f"Top 2 fitness: {fitness_values[0]:.2f}, {fitness_values[1]:.2f}")
            
            # Update best agent if improved
            if best_fitness > self.best_fitness:
                self.best_fitness = best_fitness
                self.best_agent = evaluated_pop[0][0]
                tqdm.write(f"New best agent with fitness {best_fitness:.2f}!")
                
                # Save best agent if requested
                if self.save_best:
                    with open(self.save_path, "wb") as f:
                        pickle.dump(self.best_agent, f)
                    tqdm.write(f"Saved best agent to {self.save_path}")
            
            # Render best agent if requested
            if self.render_best and generation % 10 == 0:
                tqdm.write("Rendering best agent...")
                self.evaluate_agent(evaluated_pop[0][0], render=True)
            
            # Selection - only top 2 performers
            selected = self.selection(evaluated_pop)
            
            # Crossover and mutation to create next generation
            self.population = self.crossover_and_mutation(selected)
            
            # Plot progress
            if generation % 10 == 0:
                self.plot_progress()
        
        # Final evaluation of best agent
        if self.best_agent is not None:
            tqdm.write("\nFinal evaluation of best agent:")
            final_fitness = self.evaluate_agent(self.best_agent, render=self.render_best)
            tqdm.write(f"Final fitness: {final_fitness:.2f}")
        
        # Plot final progress
        self.plot_progress()
    
    def plot_progress(self):
        """Plot training progress."""
        plt.figure(figsize=(10, 6))
        x = np.arange(len(self.best_fitness_per_gen)) + 1
        plt.plot(x, self.best_fitness_per_gen, label="Best Fitness")
        plt.plot(x, self.avg_fitness_per_gen, label="Average Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True)
        plt.savefig("training_progress.png")
        plt.close()


def test_agent(agent_path="best_agent.pkl", render_mode="terminal"):
    """Test a saved agent."""
    # Load agent
    with open(agent_path, "rb") as f:
        agent = pickle.load(f)
    
    # Create environment
    env = SnakeEnvironment(width=15, height=15, render_size=30)
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    max_steps = 5000
    
    # Create progress bar for testing
    with tqdm(total=max_steps, desc="Testing agent") as pbar:
        while not done and steps < max_steps:
            env.render(mode=render_mode)
            
            # Choose action
            action = agent.act(state)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Update agent
            agent.update(reward)
            
            # Update state and total reward
            state = next_state
            total_reward += reward
            steps += 1
            
            # Update progress bar
            if "score" in info:
                score = info["score"]
                pbar.set_description(f"Testing agent (Score: {score})")
            pbar.update(1)
            
            # Control speed
            time.sleep(0.1)
    
    tqdm.write(f"Game over! Final score: {info['score']} in {steps} steps")

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Train or test a Snake agent with a BrainSNN")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "manual"],
                        help="Mode: train, test, or manual")
    parser.add_argument("--agent", type=str, default="best_agent.pkl",
                        help="Path to saved agent for testing")
    parser.add_argument("--generations", type=int, default=50,
                        help="Number of generations for training")
    parser.add_argument("--population", type=int, default=10,
                        help="Population size for training")
    parser.add_argument("--render", action="store_true",
                        help="Render the best agent during training")
    parser.add_argument("--sequential", action="store_true",
                        help="Use sequential evaluation instead of parallel")
    parser.add_argument("--processes", type=int, default=None,
                        help="Number of processes to use for parallel evaluation")
    parser.add_argument("--render-mode", type=str, default="terminal", choices=["terminal", "human"],
                        help="Rendering mode: terminal or (human)")
    args = parser.parse_args()
    
    if args.mode == "train":
        trainer = EvolutionTrainer(
            population_size=args.population,
            num_generations=args.generations,
            render_best=args.render,
            use_parallel=not args.sequential,
            num_processes=args.processes,
            render_mode=args.render_mode
        )
        trainer.train()
    elif args.mode == "test":
        test_agent(args.agent, render_mode=args.render_mode)