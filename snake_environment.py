import numpy as np
from enum import Enum
from typing import Tuple

class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class SnakeEnvironment:
    def __init__(self, width: int = 20, height: int = 20, render_size: int = 20):
        """
        Initialize the Snake game environment.
        
        Args:
            width: Width of the game grid
            height: Height of the game grid
            render_size: Size of each cell when rendering (in pixels)
        """
        self.width = width
        self.height = height
        self.render_size = render_size
        
        # Game state
        self.snake_body = None
        self.snake_direction = None
        self.food_position = None
        self.game_over = None
        self.score = None
        self.steps_without_food = None
        self.max_steps_without_food = width * height  # Adjust as needed
        
        # For rendering
        self.screen = None
        self.clock = None
        self.is_rendering = False
        
        # Ray directions for vision (normalized)
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        self.ray_directions = np.column_stack((np.cos(angles), np.sin(angles)))
        
        # Reset the environment
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        # Initialize snake in the middle
        start_x, start_y = self.width // 2, self.height // 2
        self.snake_body = [(start_x, start_y)]
        self.snake_direction = Direction.RIGHT
        
        # Place food randomly
        self._place_food()
        
        # Reset game state
        self.game_over = False
        self.score = 0
        self.steps_without_food = 0
        
        return self._get_state()
    
    def _place_food(self):
        """Place food at a random empty position."""
        if len(self.snake_body) >= self.width * self.height:
            # No empty space for food
            self.food_position = None
            return
        
        # Create all possible positions
        all_positions = [(x, y) for x in range(self.width) for y in range(self.height)]
        empty_positions = [pos for pos in all_positions if pos not in self.snake_body]
        
        # Place food at a random empty position
        if empty_positions:
            self.food_position = empty_positions[np.random.randint(0, len(empty_positions))]
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take a step in the environment with the given action.
        
        Args:
            action: Integer representing the action (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT)
            
        Returns:
            state: New state after taking the action
            reward: Reward received
            done: Whether the episode is done
            info: Additional information
        """
        # Convert action to direction
        self.snake_direction = Direction(action)
        
        # Move snake in the chosen direction
        head_x, head_y = self.snake_body[0]
        
        # Calculate new head position
        if self.snake_direction == Direction.UP:
            new_head = (head_x, head_y - 1)
        elif self.snake_direction == Direction.RIGHT:
            new_head = (head_x + 1, head_y)
        elif self.snake_direction == Direction.DOWN:
            new_head = (head_x, head_y + 1)
        elif self.snake_direction == Direction.LEFT:
            new_head = (head_x - 1, head_y)
        
        # Add new head to snake body
        self.snake_body.insert(0, new_head)
        
        # Calculate reward and check for game over conditions
        reward = 0.01  # Small reward for surviving
        self.steps_without_food += 1
        
        # Check if snake has hit the wall
        head_x, head_y = self.snake_body[0]
        if head_x < 0 or head_x >= self.width or head_y < 0 or head_y >= self.height:
            self.game_over = True
            reward = -1.0
            return self._get_state(), reward, self.game_over, {"score": self.score}
        
        # Check if snake has hit itself
        if self.snake_body[0] in self.snake_body[1:]:
            self.game_over = True
            reward = -1.0
            return self._get_state(), reward, self.game_over, {"score": self.score}
        
        # Check if snake has eaten food
        if self.food_position and self.snake_body[0] == self.food_position:
            # Reward for eating food
            reward = 1.0
            self.score += 1
            self.steps_without_food = 0
            
            # Place new food
            self._place_food()
        else:
            # Remove tail (snake didn't grow)
            self.snake_body.pop()
            
            # Calculate distance-based reward
            if self.food_position:
                # Get current and previous distance to food
                head_x, head_y = self.snake_body[0]
                food_x, food_y = self.food_position
                current_distance = abs(head_x - food_x) + abs(head_y - food_y)  # Manhattan distance
                
                # Get previous head position
                if len(self.snake_body) > 1:
                    # Use direction to calculate previous position
                    if self.snake_direction == Direction.UP:
                        prev_x, prev_y = head_x, head_y + 1
                    elif self.snake_direction == Direction.RIGHT:
                        prev_x, prev_y = head_x - 1, head_y
                    elif self.snake_direction == Direction.DOWN:
                        prev_x, prev_y = head_x, head_y - 1
                    elif self.snake_direction == Direction.LEFT:
                        prev_x, prev_y = head_x + 1, head_y
                    
                    prev_distance = abs(prev_x - food_x) + abs(prev_y - food_y)
                    
                    # If moved closer to food, give positive reward
                    if current_distance < prev_distance:
                        reward = 0.05  # Stronger reward for moving toward food
                    # If moved away from food, give small negative reward
                    elif current_distance > prev_distance:
                        reward = -0.01  # Reduced penalty
        
        # Check if snake is stuck in a loop
        if self.steps_without_food >= self.max_steps_without_food:
            self.game_over = True
            reward = -0.5
        
        return self._get_state(), reward, self.game_over, {"score": self.score}
    
    def _get_state(self) -> np.ndarray:
        """
        Get the current state observation as a binary numpy array.
        
        Returns:
            state: Game state as a numpy array with vision data
        """
        # Initialize an empty state
        state = {}
        
        # Get current direction as one-hot encoding
        direction_one_hot = np.zeros(4, dtype=np.float32)
        direction_one_hot[self.snake_direction.value] = 1
        state["direction"] = direction_one_hot
        
        # Get immediate danger signals
        danger = self._get_danger_signals()
        state["danger"] = danger
        
        # Get vision ray information
        vision = self._get_vision_rays()
        state["vision"] = vision.flatten()
        
        # Get food direction
        food_direction = self._get_food_direction()
        state["food_direction"] = food_direction
        
        return state
    
    def _get_danger_signals(self) -> np.ndarray:
        """
        Get immediate danger signals for the three directions relative to the snake's head.
        
        Returns:
            danger: Binary array [forward, right, left] indicating danger
        """
        # Calculate relative directions
        head_direction = self.snake_direction.value
        forward = head_direction
        right = (head_direction + 1) % 4
        left = (head_direction - 1) % 4
        
        # Convert directions to coordinate offsets
        direction_to_offset = {
            Direction.UP.value: (0, -1),
            Direction.RIGHT.value: (1, 0),
            Direction.DOWN.value: (0, 1),
            Direction.LEFT.value: (-1, 0)
        }
        
        # Get head position
        head_x, head_y = self.snake_body[0]
        
        # Check danger in three directions
        danger = np.zeros(3, dtype=np.float32)
        
        # Check forward
        dx, dy = direction_to_offset[forward]
        check_x, check_y = head_x + dx, head_y + dy
        if (check_x < 0 or check_x >= self.width or
            check_y < 0 or check_y >= self.height or
            (check_x, check_y) in self.snake_body):
            danger[0] = 1
        
        # Check right
        dx, dy = direction_to_offset[right]
        check_x, check_y = head_x + dx, head_y + dy
        if (check_x < 0 or check_x >= self.width or
            check_y < 0 or check_y >= self.height or
            (check_x, check_y) in self.snake_body):
            danger[1] = 1
        
        # Check left
        dx, dy = direction_to_offset[left]
        check_x, check_y = head_x + dx, head_y + dy
        if (check_x < 0 or check_x >= self.width or
            check_y < 0 or check_y >= self.height or
            (check_x, check_y) in self.snake_body):
            danger[2] = 1
        
        return danger
    
    def _get_vision_rays(self) -> np.ndarray:
        """
        Cast rays in 8 directions to detect walls, food, and snake body.
        
        Returns:
            vision: Array of shape (8, 3) containing normalized distances
                   [wall_distance, food_distance, body_distance] for each ray
        """
        head_x, head_y = self.snake_body[0]
        head_pos = np.array([head_x, head_y], dtype=np.float32)
        
        # Calculate maximum possible distance for normalization
        max_distance = np.sqrt(self.width**2 + self.height**2)
        
        # Initialize vision array
        num_rays = len(self.ray_directions)
        vision = np.ones((num_rays, 3), dtype=np.float32)  # [wall, food, body]
        
        # Cast rays
        for i, direction in enumerate(self.ray_directions):
            # Initialize distances to max
            wall_distance = max_distance
            food_distance = max_distance
            body_distance = max_distance
            
            # Start from head position
            current_pos = head_pos.copy()
            
            # Cast ray until hitting wall
            for step in range(1, int(max_distance) + 1):
                # Move one step in the ray direction
                current_pos += direction
                current_x, current_y = int(current_pos[0]), int(current_pos[1])
                
                # Check if hit wall
                if (current_x < 0 or current_x >= self.width or
                    current_y < 0 or current_y >= self.height):
                    wall_distance = step
                    break
                
                # Check if hit food
                if self.food_position and (current_x, current_y) == self.food_position:
                    food_distance = step
                
                # Check if hit snake body
                if (current_x, current_y) in self.snake_body:
                    body_distance = step
            
            # Normalize distances
            vision[i, 0] = wall_distance / max_distance
            vision[i, 1] = food_distance / max_distance
            vision[i, 2] = body_distance / max_distance
        
        return vision
    
    def _get_food_direction(self) -> np.ndarray:
        """
        Get the direction to food as a normalized vector.
        
        Returns:
            food_direction: Normalized vector pointing toward food
        """
        if not self.food_position:
            return np.zeros(2, dtype=np.float32)
        
        head_x, head_y = self.snake_body[0]
        food_x, food_y = self.food_position
        
        # Calculate direction vector
        dx = food_x - head_x
        dy = food_y - head_y
        
        # Normalize
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            dx /= length
            dy /= length
        
        return np.array([dx, dy], dtype=np.float32)
    
    def render(self, mode: str = 'human'):
        """
        Render the game state.
        
        Args:
            mode: 'human' for pygame visualization, 'terminal' for console output
        """
        if mode == 'terminal':
            self._render_terminal()
            return
            
                
    def _render_terminal(self):
        """Render the game state in the terminal."""
        # Clear terminal (cross-platform)
        print("\033[H\033[J", end="")
        
        # Create grid
        grid = [['·' for _ in range(self.width)] for _ in range(self.height)]
        
        # Add food
        if self.food_position:
            x, y = self.food_position
            grid[y][x] = 'F'
        
        # Add snake
        for i, (x, y) in enumerate(self.snake_body):
            if i == 0:
                # Head
                if self.snake_direction == Direction.UP:
                    grid[y][x] = '^'
                elif self.snake_direction == Direction.RIGHT:
                    grid[y][x] = '>'
                elif self.snake_direction == Direction.DOWN:
                    grid[y][x] = 'v'
                elif self.snake_direction == Direction.LEFT:
                    grid[y][x] = '<'
            else:
                # Body
                grid[y][x] = 'O'
        
        # Print grid
        border = '+' + '-' * self.width + '+'
        print(border)
        for row in grid:
            print('|' + ''.join(row) + '|')
        print(border)
        
        # Print game info
        print(f"Score: {self.score} | Direction: {self.snake_direction.name} | Length: {len(self.snake_body)}")
        print(f"Steps without food: {self.steps_without_food}/{self.max_steps_without_food}")
    