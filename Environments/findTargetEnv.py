from typing import Any, SupportsFloat
import target as tg
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType, ActType, RenderFrame

from rendering import Renderer
import numpy as np
import pygame


class FindTargetEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size:int = 5, render_mode=None):
        self.size = size
        # define the action space
        self.action_space = spaces.Discrete(4)
        # The actions are mapped to left, right, up, down
        self.action_to_direction = {
            0: np.array([0,1]), 
            1: np.array([0, -1]), 
            2: np.array([1,0]), 
            3: np.array([-1, 0])}
        
        #observation space
        observation_shape = (2,)
        self.observation_space = spaces.Box(low=0, 
                                            high=size-1, 
                                            shape=observation_shape, 
                                            dtype=int)

        # rendering
        self.renderer = Renderer(meta_data=self.metadata, grid_size=self.size, render_mode=render_mode)
        self._set_up()
        self._new_episode = False
        self.renderer = Renderer(
            meta_data=self.metadata, grid_size=self.size, render_mode=render_mode
        )

    def _set_up(self):
        """
        Setup the environment
        :return:
        """
        self._setup_targets()

        # Randomly sample the agent's location until it does not match any of the target's locations.
        self._agent_location = self._targets[0].position
        positions = [target.position for target in self._targets]
        while np.any(np.all(self._agent_location == positions, axis=1)):
            self._agent_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )
        
        # rendering
        self._counted_positions = {}
        self._new_episode = True


    def _setup_targets(self):
        """
        Setup the targets in the environment
        :return:
        """
        target = tg.Target(
            color= (255,0,0),
            reward = 10,
            position= np.array([2,3]),
            velocity=0.0,
            movement=tg.Direction(tg.DirectionType.NONE)
        )
        self._targets = [target]

    def _get_obs(self):
        """
        The agent's observation.
        Must match the observation space defined in init
        :return: The agent's observation'
        """
        return self._agent_location
    
    def _get_info(self):
        """
        Info dictionary for gymnasium.
        This is not used for the RL training
        :return: Info dictionary
        """
        return {
            "distance": int(np.linalg.norm(self._agent_location - self._targets[0].position, ord=1))
        }
    
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        # your code here

        self._render_frame_for_humans_if_needed()
        
        return obs, info
    
    def _get_new_agent_position_from_action(self, action: ActType):
        new_pos = self._agent_location + self.action_to_direction[action]
        # Check if the new position is on the grid, if not return the old one
        if np.any(new_pos < np.zeros((2,), dtype=int)) or np.any(new_pos > np.full((2,), fill_value=self.size-1)):
            return self._agent_location
        return new_pos
    
    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # rendering
        self._new_episode = False
        self._count_position(position=(self._agent_location[0], self._agent_location[1]))
        
        # your code here



        self._render_frame_for_humans_if_needed()
        
        return obs, reward, terminated, False, info
        

    # rendering

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        self.renderer.render(agent_location=self._agent_location, new_episode=self._new_episode, targets=self._targets,
                             visited_cells_count=self._counted_positions)

    def _render_frame(self):
        return self.renderer.render_frame(
            agent_location=self._agent_location,
            new_episode=self._new_episode,
            targets=self._targets,
            visited_cells_count=self._counted_positions,
        )

    def _render_frame_for_humans_if_needed(self):
        return self.renderer.render_frame_for_humans_if_needed(
            agent_location=self._agent_location,
            new_episode=self._new_episode,
            targets=self._targets,
            visited_cells_count=self._counted_positions,
        )

    def _count_position(self, position: tuple[int, int]):
        try:
            self._counted_positions[position] += 1
        except:
            self._counted_positions[position] = 1

    def close(self):
        if self.renderer.window is not None:
            pygame.display.quit()
            pygame.quit()