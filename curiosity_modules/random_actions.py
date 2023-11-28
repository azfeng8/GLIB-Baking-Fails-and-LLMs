"""Curiosity module that just takes random actions.
"""

from curiosity_modules import BaseCuriosityModule


class RandomCuriosityModule(BaseCuriosityModule):
    """Curiosity module that just takes random actions."""

    def _initialize(self):
        pass

    def reset_episode(self, state):
        pass

    def get_action(self, state, iter_path=None):
        return self._action_space.sample(state)
