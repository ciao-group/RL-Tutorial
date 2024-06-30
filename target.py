import numpy as np
from enum import Enum
from typing import Tuple, Callable


class DirectionType(Enum):
    """
    Enumeration representing different movement directions.

    Attributes:
        NONE (int): No movement.
        UP (int): Upward movement.
        DOWN (int): Downward movement.
        LEFT (int): Leftward movement.
        RIGHT (int): Rightward movement.
        DIAGONAL_LEFT_DOWN (int): Diagonal movement to the left and down.
        DIAGONAL_LEFT_UP (int): Diagonal movement to the left and up.
        DIAGONAL_RIGHT_DOWN (int): Diagonal movement to the right and down.
        DIAGONAL_RIGHT_UP (int): Diagonal movement to the right and up.
        APPEAR (int): Represents appearing or no change in position.
    """

    NONE = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    DIAGONAL_LEFT_DOWN = 5
    DIAGONAL_LEFT_UP = 6
    DIAGONAL_RIGHT_DOWN = 7
    DIAGONAL_RIGHT_UP = 8
    APPEAR = 9


class Direction:
    """
    Class representing a specific movement direction.

    Attributes:
        direction_type (DirectionType): The type of movement direction.

    Methods:
        reverse(): Reverse the current direction.
        movement_vector() -> np.ndarray: Get the movement vector associated with the direction.
    """

    def __init__(self, direction_type: DirectionType):
        """
        Initialize a Direction object with a given direction type.

        Parameters:
            direction_type (DirectionType): The initial direction type.
        """
        self.direction_type = direction_type

    def reverse(self):
        """
        Reverse the current direction.
        """
        match self.direction_type:
            case DirectionType.UP:
                self.direction_type = DirectionType.DOWN
            case DirectionType.DOWN:
                self.direction_type = DirectionType.UP
            case DirectionType.LEFT:
                self.direction_type = DirectionType.RIGHT
            case DirectionType.RIGHT:
                self.direction_type = DirectionType.LEFT
            case DirectionType.DIAGONAL_LEFT_DOWN:
                self.direction_type = DirectionType.DIAGONAL_RIGHT_UP
            case DirectionType.DIAGONAL_LEFT_UP:
                self.direction_type = DirectionType.DIAGONAL_RIGHT_DOWN
            case DirectionType.DIAGONAL_RIGHT_DOWN:
                self.direction_type = DirectionType.DIAGONAL_LEFT_UP
            case DirectionType.DIAGONAL_RIGHT_UP:
                self.direction_type = DirectionType.DIAGONAL_LEFT_DOWN
            case _:
                self.direction_type = DirectionType.NONE

    def movement_vector(self):
        """
        Get the movement vector associated with the direction.

        Returns:
            np.ndarray: A NumPy array representing the movement vector.
                        For example, np.array([1, 0]) for rightward movement.
        """
        match self.direction_type:
            case DirectionType.UP:
                return np.array([0, -1])
            case DirectionType.DOWN:
                return np.array([0, 1])
            case DirectionType.RIGHT:
                return np.array([1, 0])
            case DirectionType.LEFT:
                return np.array([-1, 0])
            case DirectionType.DIAGONAL_LEFT_UP:
                return np.array([-1, -1])
            case DirectionType.DIAGONAL_LEFT_DOWN:
                return np.array([-1, 1])
            case DirectionType.DIAGONAL_RIGHT_UP:
                return np.array([1, -1])
            case DirectionType.DIAGONAL_RIGHT_DOWN:
                return np.array([1, 1])
            case DirectionType.NONE | DirectionType.APPEAR:
                return np.array([0, 0])
            case _:
                return np.array([0, 0])

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.direction_type.name == other.direction_type.name
        return False



class Target:
    """
    Class representing a target.

    Attributes:
        color (tuple): RGB color tuple representing the target's color.
        reward (int): Reward associated with the target.
        position (np.ndarray): Current position of the target as a NumPy array.
        velocity (float): Velocity of the target.
        movement (Direction): Object representing the target's movement direction.
        _org_position (np.ndarray): Original position of the target.
        steps_per_timestep (int): Number of steps the target moves in a single timestep.
        steps_until_next_steps (int): Steps remaining until the target moves again.

    Methods:
        step(self): Move the target according to its configuration for one timestep.
        reverse_direction(self): Reverse the direction of the target's movement.
        update_position(self, new_position: np.ndarray): Update the target's position.
        is_hit(self, pos_to_compare: np.ndarray) -> bool: Check if a given position matches the target's position.

        @staticmethod
        dummy_target() -> Target: Create a dummy target with default properties.

    """

    def __init__(
        self,
        color,
        reward: int,
        position: np.ndarray,
        velocity: float,
        movement: Direction,
        random_start: bool = False,
    ):
        """
        Initialize a Target object.

        Parameters:
            color (tuple): RGB color tuple representing the target's color.
            reward (int): Reward associated with the target.
            position (np.ndarray): Initial position of the target as a NumPy array.
            velocity (float): Velocity of the target.
            movement (Direction): Object representing the target's movement direction.
            random_start (bool): Flag indicating if the target spawns randomly.
            random_start_range (list(Tuple[int, int])): The range of x and y coordinates where the agent can spawn.
            random_moving: Flag indicating if the target moves randomly (forward / backward) along the specified movement type
        """

        self.color = color
        self.reward = reward
        self.position = position
        self.velocity = velocity
        self.movement = movement
        self.random_start = random_start

        self._org_position = position

        if velocity <= 0:
            self.steps_per_timestep = 0
            self.steps_until_next_steps = 0
            return

        velocity = 1 if np.isinf(velocity) else velocity

        self.steps_per_timestep = int(np.round(velocity)) if velocity >= 1 else 1
        self.steps_until_next_steps = 0 if velocity >= 1 else np.round(1 / velocity) - 1

    def step(self, elapsed_time: float = None):
        """
        Move the target according to its configuration for one timestep.
        :param elapsed_time (optional): The time in seconds elapsed in the environment, this is used to calculate if the target changes its direction
        """

        if self.steps_until_next_steps == 0:
            for step in range(self.steps_per_timestep):
                if self.movement.direction_type == DirectionType.APPEAR:
                    if self.random_start:
                            self.set_random_position()
                    else:
                        self.position = (
                            np.array([-1, -1])
                            if np.array_equal(self.position, self._org_position)
                            else self._org_position
                        )
                else:
                    self.position += self.movement.movement_vector()
                self.steps_until_next_steps = (
                    0 if self.velocity >= 1 else np.round(1 / self.velocity) - 1
                )
        else:
            self.steps_until_next_steps -= 1

    def reverse_direction(self):
        """
        Reverse the direction of the target's movement.
        """
        self.movement.reverse()

    def update_position(self, new_position: np.array):
        """
        Update the target's position.

        Parameters:
            new_position (np.ndarray): New position of the target as a NumPy array.
        """
        self.position = new_position

