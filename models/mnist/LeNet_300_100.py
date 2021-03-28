# Copyright 2019 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch.nn as nn
import torch.nn.functional as F


class LeNet_300_100(nn.Module):
  """Simple NN with hidden layers [300, 100]
  Based on https://github.com/mi-lad/snip/blob/master/train.py
  by Milad Alizadeh.
  """

  def __init__(self):
    super(LeNet_300_100, self).__init__()
    self.fc1 = nn.Linear(28 * 28 * 1, 300, bias=True)
    self.fc2 = nn.Linear(300, 100, bias=True)
    self.fc3 = nn.Linear(100, 10, bias=True)
    self.mask = None

  def forward(self, x):
    x0 = x.view(-1, 28 * 28 * 1)
    x1 = F.relu(self.fc1(x0))
    x2 = F.relu(self.fc2(x1))
    x3 = self.fc3(x2)
    return F.log_softmax(x3, dim=1)