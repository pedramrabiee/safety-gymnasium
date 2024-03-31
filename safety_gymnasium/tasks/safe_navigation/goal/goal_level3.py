# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Goal level 3."""

from safety_gymnasium.assets.free_geoms import Vases
from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.assets.geoms import RectHazards
from safety_gymnasium.assets.geoms import Walls
from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0


class GoalLevel3(GoalLevel0):
    """An agent must navigate to a goal while avoiding hazards.

    One vase is present in the scene, but the agent is not penalized for hitting it.
    """

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.placements_conf.extents = [-2.5, -2.5, 2.5, 2.5]

        self._add_geoms(Hazards(num=3, size=0.2, keepout=0.18, name='0_hazards'))
        self._add_geoms(Hazards(num=2, size=0.3, keepout=0.6, name='1_hazards'))
        self._add_geoms(RectHazards(num=1, keepout=0.6, name='0_rect_hazards'))
        self._add_geoms(RectHazards(num=2, size=[0.2, 0.6], keepout=0.6, name='1_rect_hazards'))

        # self._add_geoms(Walls(num=1))
        # self._add_free_geoms(Vases(num=1, is_constrained=False))
