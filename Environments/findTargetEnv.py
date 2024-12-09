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
                                            dtype=integer)
        

        # rendering
        self.renderer = Renderer(meta_data=self.metadata, grid_size=self.size, render_mode=render_mode)
        self._set_up()
        self._new_episode = False
        self.renderer = Renderer(
            meta_data=self.metadata, grid_size=self.size, render_mode=render_mode
        )

    def _set_up(self):
        self._agent_location = np.array([0,0])
        self._setup_targets()
        # rendering
        self._counted_positions = {}
        self._new_episode = True

    def _setup_targets(self):
        target = tg.Target(
            color= (255,0,0),
            reward = 10,
            position= np.array([2,3]),
            velocity=0.0,
            movement=tg.Direction(tg.DirectionType.NONE)
        )
        self._targets = [target]

    def _get_obs(self):
        return self._agent_location
    
    def _get_info(self):
        return {
            #"distance": np.linalg.norm(self._agent_location - self._targets[0].position)
            "distance": 0
        }

    
    def _get_new_agent_position_from_action(self, action: ActType):
        new_pos = self._agent_location + self.action_to_direction[action]  
        if np.any(new_pos < np.zeros((2,), dtype=int)) \
            or np.any(new_pos > np.full((2,), fill_value=self.size-1)):
            return self._agent_location
        return new_pos
        
    

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