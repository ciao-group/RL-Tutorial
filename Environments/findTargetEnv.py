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
        pass




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