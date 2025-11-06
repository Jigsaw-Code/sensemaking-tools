# Copyright 2025 Google LLC
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

import unittest
from unittest.mock import AsyncMock

from models import vertex_model
from models.vertex_model import TokenLimitExceededError


class VertexModelTest(unittest.IsolatedAsyncioTestCase):

  async def test_retry_call_token_limit_exceeded(self):
    mock_func = AsyncMock()
    mock_func.side_effect = TokenLimitExceededError(
        "exceeds the maximum number of tokens allowed"
    )

    with self.assertRaises(TokenLimitExceededError) as context:
      await vertex_model._retry_call(mock_func, lambda x: True, 3, "error", 1)

    self.assertIn(
        "exceeds the maximum number of tokens allowed", str(context.exception)
    )
    self.assertEqual(mock_func.call_count, 1)


if __name__ == "__main__":
  unittest.main()
