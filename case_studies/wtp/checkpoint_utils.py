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

"""Utility functions for checkpointing."""

import logging
import os
import pickle
from typing import Any, Optional


def _get_checkpoint_dir(output_dir: str) -> str:
  """Returns the path to the checkpoint directory."""
  return os.path.join(output_dir, ".checkpoints")


def _get_checkpoint_path(output_dir: str, step_name: str) -> str:
  """Returns the path to a specific checkpoint file."""
  return os.path.join(_get_checkpoint_dir(output_dir), f"{step_name}.pkl")


def save_checkpoint(data: Any, step_name: str, output_dir: Optional[str]):
  """Saves data to a checkpoint file.

  Args:
      data: The data to save.
      step_name: The name of the step being checkpointed.
      output_dir: The output directory. If None, checkpointing is skipped.
  """
  if not output_dir:
    logging.debug(
        f"Skipping save checkpoint for '{step_name}': output_dir is None."
    )
    return
  checkpoint_dir = _get_checkpoint_dir(output_dir)
  os.makedirs(checkpoint_dir, exist_ok=True)
  checkpoint_path = _get_checkpoint_path(output_dir, step_name)
  logging.info(f"Saving checkpoint for step '{step_name}' to {checkpoint_path}")
  with open(checkpoint_path, "wb") as f:
    pickle.dump(data, f)  # type: ignore


def load_checkpoint(step_name: str, output_dir: Optional[str]) -> Optional[Any]:
  """Loads data from a checkpoint file if it exists.

  Args:
      step_name: The name of the step to load.
      output_dir: The output directory. If None, checkpointing is skipped.

  Returns:
      The loaded data, or None if the checkpoint does not exist or output_dir is
      None.
  """
  if not output_dir:
    logging.debug(
        f"Skipping load checkpoint for '{step_name}': output_dir is None."
    )
    return None
  checkpoint_path = _get_checkpoint_path(output_dir, step_name)
  if not os.path.exists(checkpoint_path):
    logging.info(
        f"No checkpoint found for step '{step_name}' at {checkpoint_path}"
    )
    return None
  logging.info(
      f"Loading checkpoint for step '{step_name}' from {checkpoint_path}"
  )
  try:
    with open(checkpoint_path, "rb") as f:
      return pickle.load(f)
  except Exception as e:
    logging.warning(
        f"Failed to load checkpoint for '{step_name}': {e}. "
        "Ignoring and treating as missing."
    )
    return None
