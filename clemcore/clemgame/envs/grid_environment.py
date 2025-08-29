import io
import logging
from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TypedDict

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from clemcore.clemgame.envs.environment import GameEnvironment, GameState
from clemcore.clemgame.player import Player

logger = logging.getLogger(__name__)

Position = Tuple[int, int]


@dataclass
class Object(ABC):
    """Base class for all objects in the grid environment."""
    position: Position
    name: str
    symbol: str  # char to be shown in the grid
    pretty_symbol: str  # emoji to be shown in the grid on render_state_as_human_readable

    def __str__(self) -> str:
        return f"{self.name} at {self.position}"


class GridCell(TypedDict):
    objects: List[Object]
    position: Position


Grid = list[list[GridCell]]


class GridState(GameState):
    """Extended game state for grid-based environments.

    Additional fields:
    - _grid: The 2D grid of objects
    - _player_positions: Optional dictionary mapping player names to their positions (only required if players should be positioned in the grid)
    - _explored: Optional dictionary mapping player names to their explored part of the grid (only required if show_explored is True)
    """
    _grid: Grid
    _player_positions: Optional[Dict[str, Position]]
    _explored: Optional[Dict[str, List[List[bool]]]]


class PlayerObject(Object):
    """Optionally represents a player in the grid."""

    def __init__(self, position: Position, player: Player):
        super().__init__(position, f"Player_{player.name}", "P", "👤")
        self.player = player


class GridEnvironment(GameEnvironment):
    """Base class for grid-based game environments."""

    def __init__(
        self,
        config: Optional[Dict] = None,
    ):
        """Initialize the grid environment.

        Args:
            config: Additional configuration options
        """
        super().__init__(config)

        self.width = config.get("width", 10)
        self.height = config.get("height", 10)
        self.limited_visibility = config.get("limited_visibility", False)
        self.show_explored = config.get("show_explored", False)

        self.state: GridState = {
            "_grid": [
                [GridCell(objects=[], position=(y, x)) for x in range(self.width)]
                for y in range(self.height)
            ],
            "_player_positions": None,
            "_explored": None,
        }

    def reset(self):
        """Reset the environment to its initial state."""
        super().reset()

        self.state["_grid"] = [[GridCell(objects=[], position=(y, x))
                                        for x in range(self.width)] for y in range(self.height)]

        # write players_start=None in instance generator if players should not be
        # positioned in the grid, otherwise [(row, col), (row, col), ...]
        players_start = self.config.get("grid", {}).get("players_start", None)
        if players_start:
            self.state["_player_positions"] = {}
            for i, player in enumerate(self.players):
                player_start = players_start[i]
                self.add_object(PlayerObject(position=player_start, player=player))
                self.state["_player_positions"][player.name] = tuple(player_start)

        if self.show_explored:
            self.state["_explored"] = {}
            for player in self.players:
                self.state["_explored"][player.name] = [
                    [False for _ in range(self.width)] for _ in range(self.height)
                ]
                self._mark_explored(player.name, self.get_player_position(player.name))

    def _mark_explored(self, player_name: str, pos: Position) -> None:
        """Mark cells around a position as explored for the given player."""
        row, col = pos
        for i in range(row - 1, row + 2):
            for j in range(col - 1, col + 2):
                if 0 <= i < self.height and 0 <= j < self.width:
                    self.state["_explored"][player_name][i][j] = True

    def add_object(self, obj: Object) -> None:
        """Add an object to the grid at its position."""
        y, x = obj.position
        if 0 <= x < self.width and 0 <= y < self.height:
            self.state["_grid"][y][x]["objects"].append(obj)
        else:
            raise ValueError(f"Position {obj.position} is out of bounds")

    def remove_object(self, obj: Object) -> None:
        """Remove an object from the grid."""
        y, x = obj.position
        if obj in self.state["_grid"][y][x]["objects"]:
            self.state["_grid"][y][x]["objects"].remove(obj)

    def get_objects_at(self, position: Position) -> List[Object]:
        """Get all objects at a given position."""
        y, x = position
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.state["_grid"][y][x]["objects"]
        return []

    def get_player_position(self, player_name: str) -> Position:
        """Get the position of a player."""
        return self.state["_player_positions"][player_name]

    def get_player_object(self, player_name: str) -> PlayerObject:
        """Get the player object for a given player."""
        return self.get_objects_at(self.get_player_position(player_name))[-1]

    def move_player(self, player_name: str, direction: str) -> None:
        """Move a player in a given direction."""
        y, x = self.get_player_position(player_name)
        player_object = self.get_player_object(player_name)
        self.remove_object(player_object)

        if direction == "n":
            player_object.position = (y - 1, x)
        elif direction == "s":
            player_object.position = (y + 1, x)
        elif direction == "e":
            player_object.position = (y, x + 1)
        elif direction == "w":
            player_object.position = (y, x - 1)

        self.add_object(player_object)

        if self.show_explored:
            self._mark_explored(player_name, player_object.position)

        self.state["_player_positions"][player_name] = player_object.position

    def _is_action_valid_in_state(
        self, player: Player, direction: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Basic check if a move is valid, assuming the player is part of the grid and the action is to move."""
        if direction is None:
            # no abstract rules for actions other than moving
            return True, ""

        y, x = self.get_player_position(player.name)

        if direction == "n":
            new_pos = (y - 1, x)
        elif direction == "s":
            new_pos = (y + 1, x)
        elif direction == "e":
            new_pos = (y, x + 1)
        elif direction == "w":
            new_pos = (y, x - 1)
        else:
            return False, f"Invalid direction: {direction}! Please try again."

        new_y, new_x = new_pos
        # check if the new position is within the grid
        if not (0 <= new_y < self.height and 0 <= new_x < self.width):
            return (
                False,
                f"The cell ({new_y}, {new_x}) is outside the grid! Please try again.",
            )

        return True, ""

    def _render_state_as_string(self, player_name: Optional[str] = None) -> str:
        """Format the grid for display as string."""
        grid_str = ""

        player_pos = None
        explored = None

        if player_name is not None and self.state["_player_positions"] is not None and self.show_explored:
            player_pos = self.state["_player_positions"][player_name]
            explored = self.state["_explored"][player_name]

        # render visible area of player if limited visibility is enabled
        if player_pos is not None and self.limited_visibility:
            y, x = player_pos
            for i in range(max(0, y - 1), min(self.height, y + 2)):
                row_str = ""
                for j in range(max(0, x - 1), min(self.width, x + 2)):
                    cell = self.state["_grid"][i][j]
                    cell_content = cell["objects"][-1].symbol if cell["objects"] != [] else "empty"
                    row_str += f"({i},{j}) is {cell_content}, "
                grid_str += row_str.lstrip() + "\n"
            return grid_str

        # render full grid
        for i in range(self.height):
            row_str = ""
            for j in range(self.width):
                cell = self.state["_grid"][i][j]
                if explored is not None:
                    if explored[i][j]:
                        cell_content = cell["objects"][-1].symbol if cell["objects"] != [] else "empty"
                    else:
                        cell_content = "❓"
                else:
                    cell_content = cell["objects"][-1].symbol if cell["objects"] != [] else "empty"
                row_str += f"({i},{j}) is {cell_content}, "
            grid_str += row_str.lstrip() + "\n"
        return grid_str

    def _render_state_as_image(self, player_name: Optional[str] = None) -> bytes:
        """Format the grid for display as image.

        Args:
            player_name: Optional player name. If provided, uses the explored map of that player
                to render explored vs unexplored cells and marks the player's current position with 'player'.
                If None, shows the entire grid without fog of war.

        Returns:
            Base64-encoded PNG image data
        """
        fig, ax = plt.subplots(figsize=(max(6, self.width * 0.8), max(4, self.height * 0.6)))
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_facecolor('white')

        player_pos = None
        explored = None
        if player_name is not None:
            player_pos = self.state["_player_positions"][player_name]
            explored = self.state["_explored"][player_name]

        for i in range(self.height):
            for j in range(self.width):
                cell = self.state["_grid"][i][j]

                if explored is not None and not explored[i][j]:
                    cell_content = "?"
                    cell_color = 'lightgray'
                else:
                    if cell["objects"]:
                        cell_content = cell["objects"][-1].symbol
                        if isinstance(cell["objects"][-1], PlayerObject):
                            cell_color = 'lightblue'
                        else:
                            cell_color = 'lightgreen'
                    else:
                        cell_content = " "
                        cell_color = 'white'

                rect = patches.Rectangle((j, self.height - 1 - i), 1, 1,
                                         linewidth=1, edgecolor='black',
                                         facecolor=cell_color)
                ax.add_patch(rect)

                ax.text(j + 0.5, self.height - 1 - i + 0.5, cell_content,
                        ha='center', va='center', fontsize=16, fontweight='bold')

        if player_pos is not None:
            row, col = player_pos
            rect = patches.Rectangle((col, self.height - 1 - row), 1, 1,
                                     linewidth=3, edgecolor='red',
                                     facecolor='none')
            ax.add_patch(rect)

        # if limited visibility, darken cells outside visible range
        if self.limited_visibility and player_pos is not None:
            row, col = player_pos
            for i in range(self.height):
                for j in range(self.width):
                    if abs(i - row) > 1 or abs(j - col) > 1:
                        rect = patches.Rectangle((j, self.height - 1 - i), 1, 1,
                                                 linewidth=1, edgecolor='black',
                                                 facecolor='darkgray', alpha=0.7)
                        ax.add_patch(rect)

        plt.tight_layout()

        # convert to base64-encoded PNG
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plt.close(fig)

        return buffer.getvalue()

    def _render_state_as_human_readable(self, player_name: Optional[str] = None) -> str:
        """
        Pretty print the grid state.
        """
        pretty_grid = ""
        for row in self.state["_grid"]:
            row_str = ""
            for cell in row:
                row_str += f"{cell['objects'][-1].pretty_symbol if cell['objects'] != [] else '⬜️'}"
            pretty_grid += row_str.lstrip() + "\n"
        return f"{pretty_grid}"
