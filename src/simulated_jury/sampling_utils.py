import pandas as pd
from typing import Optional


def apply_jury_size_sampling(
    df: pd.DataFrame,
    jury_size: Optional[float],
    random_state: Optional[int] = None,
    verbose: bool = False,
) -> pd.DataFrame:
  """
  Applies jury size sampling to a DataFrame.

  Args:
      df: The DataFrame to sample from.
      jury_size: The size of the jury.
                 If 0.0 < jury_size < 1.0, it's treated as a fraction.
                 If jury_size > 1.0, it's treated as an absolute number.
                 If jury_size is None or 1.0, no sampling is performed.
      random_state: Random state for reproducibility. Defaults to None (random).
      verbose: Whether to print sampling details.

  Returns:
      The sampled DataFrame.

  Raises:
      ValueError: If jury_size is <= 0.0.
  """
  if jury_size is None or jury_size == 1.0:
    return df

  if jury_size <= 0.0:
    raise ValueError(f"jury_size must be positive, got {jury_size}")

  if 0.0 < jury_size < 1.0:
    if verbose:
      print(f"\nSubsetting to {jury_size * 100:.2f}% of the original size.")
    return df.sample(frac=jury_size, random_state=random_state)

  # jury_size > 1.0
  target_size = int(jury_size)
  if verbose:
    print(f"\nSubsetting to {target_size} participants (randomly selected).")

  if target_size >= len(df):
    print(
        f"  - WARNING: Requested jury size ({target_size}) is larger than or "
        f"equal to the available participant pool ({len(df)}). Using full pool."
    )
    return df

  return df.sample(n=target_size, random_state=random_state)
