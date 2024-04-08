from safety_gymnasium.tasks.safe_navigation.goal.goal_level4 import GoalLevel4
from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.assets.geoms import RectHazards

class GoalLevel5(GoalLevel4):
    """An agent must navigate to a goal while avoiding hazards.

    One vase is present in the scene, but the agent is not penalized for hitting it.
    """

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.placements_conf.extents = [-2.5, -2.5, 2.5, 2.5]

        self._add_geoms(Hazards(num=3, size=0.2, keepout=0.18, name='0_hazards'))
        self._add_geoms(Hazards(num=2, size=0.3, keepout=0.6, name='1_hazards'))
        self._add_geoms(RectHazards(num=1, keepout=0.6, name='0_rect_hazards'))
        self._add_geoms(RectHazards(num=3, size=[0.2, 0.6], keepout=0.6, name='1_rect_hazards'))

        # self._add_geoms(Walls(num=1))
        # self._add_free_geoms(Vases(num=1, is_constrained=False))
