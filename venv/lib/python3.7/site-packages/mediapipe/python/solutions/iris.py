# Copyright 2021 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MediaPipe Iris."""

from typing import NamedTuple

import numpy as np

from mediapipe.python.solution_base import SolutionBase

BINARYPB_FILE_PATH = 'mediapipe/graphs/iris_tracking/iris_tracking_cpu.binarypb'

class Iris(SolutionBase):
  """MediaPipe Iris."""

  def __init__(self):
    super().__init__(
        binary_graph_path=BINARYPB_FILE_PATH,
        outputs=['face_landmarks_with_iris'])

  def process(self, image: np.ndarray) -> NamedTuple:
    return super().process(input_data={'input_video': image})