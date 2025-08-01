import abc
from typing import List, TYPE_CHECKING, Dict

if TYPE_CHECKING:  # to satisfy pycharm
    from clemcore.clemgame import GameMaster, GameBenchmark


class GameBenchmarkCallback(abc.ABC):

    def on_benchmark_start(self, game_benchmark: "GameBenchmark"):
        pass

    def on_game_start(self, game_master: "GameMaster", game_instance: Dict):
        pass

    def on_game_end(self, game_master: "GameMaster", game_instance: Dict):
        pass

    def on_benchmark_end(self, game_benchmark: "GameBenchmark"):
        pass


class GameBenchmarkCallbackList(GameBenchmarkCallback):

    def __init__(self, callbacks: List[GameBenchmarkCallback] = None):
        super().__init__()
        if callbacks is None:
            callbacks = []
        self.callbacks = callbacks

    def append(self, callback: GameBenchmarkCallback):
        self.callbacks.append(callback)

    def on_benchmark_start(self, game_benchmark: "GameBenchmark"):
        for callback in self.callbacks:
            callback.on_benchmark_start(game_benchmark)

    def on_game_start(self, game_master: "GameMaster", game_instance: Dict):
        for callback in self.callbacks:
            callback.on_game_start(game_master, game_instance)

    def on_game_end(self, game_master: "GameMaster", game_instance: Dict):
        for callback in self.callbacks:
            callback.on_game_end(game_master, game_instance)

    def on_benchmark_end(self, game_benchmark: "GameBenchmark"):
        for callback in self.callbacks:
            callback.on_benchmark_end(game_benchmark)
