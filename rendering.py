import pygame
import numpy as np
import sys


class Renderer:
    """
    Renders the grid world env with agent and targets.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self, grid_size: int, meta_data, window_size: int = 512, render_mode=None
    ):
        """

        :param grid_size: The size of the gird in cells (e.g. 5)
        :param meta_data: Meta data such as rendering modes and fps
        :param window_size: The displayed window size in pixels
        :param render_mode: "human", "rgb_array" or None
        """
        pygame.init()
        name = pygame.font.get_default_font()
        self._font = pygame.font.SysFont(name=name, size=20)
        self.window_size = window_size  # pyGame window size
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.grid_size = grid_size
        self.metadata = meta_data
        self.window = None
        self.clock = None

    def render(
        self,
        agent_location,
        new_episode=False,
        targets=None,
        visited_cells_count=None,
    ):
        """
        Renders in rgb_mode
        :param agent_location: The location of the agent in the grid world
        :param new_episode: Flag if a new episode has started. Renders a black screen between episodes
        :param targets: The targets of the grid world
        :param visited_cells_count: Array / Matrix that counts how often a cell (position in grid) was visited by the agent (Optional parameter.)
        """
        if self.render_mode == "rgb_array":
            return self.render_frame(
                agent_location=agent_location,
                new_episode=new_episode,
                targets=targets,
                visited_cells_count=visited_cells_count
            )

    def render_frame_for_humans_if_needed(
        self,
        agent_location,
        new_episode=False,
        targets=None,
        visited_cells_count=None
    ):
        """
        Renders the frame if rendering_mode == "human"
        :param agent_location: The location of the agent in the grid world
        :param new_episode: Flag if a new episode has started. Renders a black screen between episodes
        :param targets: The targets that should be rendered.
        :param visited_cells_count: / Matrix that counts how often a cell (position in grid) was visited by the agent (Optional parameter.)
        """
        if self.render_mode == "human":
            return self.render_frame(
                agent_location,
                new_episode=new_episode,
                targets=targets,
                visited_cells_count=visited_cells_count,
            )

    def render_frame(
        self,
        agent_location,
        new_episode=False,
        targets=None,
        visited_cells_count=None,
    ):
        """
        The actual rendering function
        :param agent_location: The location of the agent in the grid world
        :param new_episode: Flag if a new episode has started. Renders a black screen between episodes
        :param targets: The targets that should be rendered
        :param visited_cells_count: Array / Matrix that counts how often a cell (position in grid) was visited by the agent (Optional parameter.)
        """
        space_top = 0
        window_length = window_height = self.window_size

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (window_length, self.window_size + space_top)
            )

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window.get_size())
        canvas.fill((255, 255, 255))
        if new_episode:
            # render a black screen when a new episode begins
            canvas.fill((0, 0, 0))
        else:
            # the size of a single cell in pixels
            pix_square_size = self.window_size / self.grid_size

            env_grid = self.draw_environment(
                size=self.grid_size,
                pix_square_size=pix_square_size,
                agent_location=agent_location,
                targets=targets,
                visited_cells_count=visited_cells_count,
            )
            canvas.blit(env_grid, (0, space_top))


        if self.render_mode == "human":
            # copy content from canvas to visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # makes pygame window closable
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Ensure that rendering occurs at the predefined framerate
            # This code adds delay to keep framerate stable
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def draw_environment(
        self,
        size: int,
        pix_square_size,
        agent_location,
        targets=None,
        visited_cells_count=None,
    ) -> pygame.surface:
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        visited_cells_count = {} if visited_cells_count is None else visited_cells_count
        for position, times_visited in visited_cells_count.items():
            # color visited cell according to times visited
            # yellow for the first visit, then darken it linearly, darkest color is (255, 25, 0)
            red = max(255 - 2 * (times_visited - 1), 225)
            green = max(255 - 45 * (times_visited - 1), 25)
            visited_cell_color = (red, green, 0)  # yellow
            pygame.draw.rect(
                canvas,
                visited_cell_color,
                pygame.Rect(
                    pix_square_size * np.asarray(position),
                    (pix_square_size, pix_square_size),
                ),
            )
        
        if targets is None:
            targets = []

        # draw the targets
        for target in targets:
            target_color_base = (
                target.color
                if not (
                        np.array_equal(target.position, agent_location)
                )
                else (0, 255, 0)
            )
            pygame.draw.rect(
                canvas,
                target_color_base,
                pygame.Rect(
                    pix_square_size * target.position,
                    (pix_square_size, pix_square_size),
                ),
            )

        # draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # draw the grid
        self.draw_grid_lines(size=size, pix_square_size=pix_square_size, canvas=canvas)

        return canvas

    def draw_grid_lines(self, size: int, pix_square_size, canvas: pygame.surface):
        for x in range(size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )

            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )
