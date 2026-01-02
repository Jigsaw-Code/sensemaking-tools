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

# Util class for models

import os

# The maximum number of times a task should be retried.
MAX_RETRIES = 10
# The maximum number of times an LLM call should be retried.
MAX_LLM_RETRIES = 4
# How long in seconds to wait between LLM calls.
RETRY_DELAY_SEC = 10

# Set default vertex parallelism (number of concurrent LLM calls) based on similarly named env var, or use default value
parallelism_env_var = os.environ.get("DEFAULT_VERTEX_PARALLELISM")
DEFAULT_VERTEX_PARALLELISM = (
    int(parallelism_env_var) if parallelism_env_var else 1000
)
