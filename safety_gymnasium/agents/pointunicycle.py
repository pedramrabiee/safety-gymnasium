from __future__ import annotations

# from safety_gymnasium.bases.dynamics_aware_agent import DynamicsAwareAgent
from safety_gymnasium.agents.point import Point
from safety_gymnasium.agents.dynamics.unicycle_dynamics import UnicycleDynamics
from safety_gymnasium.utils.random_generator import RandomGenerator


class Pointunicycle(Point):
    def __init__(  # pylint: disable=too-many-arguments
        self,
        random_generator: RandomGenerator,
        placements: list | None = None,
        locations: list | None = None,
        keepout: float = 0.4,
        rot: float | None = None,
    ) -> None:
        # TODO: fix super
        super(Point, self).__init__(
            self.__class__.__name__,
            random_generator,
            placements,
            locations,
            keepout,
            rot,
        )
        self.dynamics = UnicycleDynamics(state_dim=4, action_dim=2)
