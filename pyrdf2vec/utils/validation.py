import os

import requests


def _check_depth(self, attribute, depth: int) -> None:
    if depth < 0:
        raise ValueError(f"'depth' must be >= 0 (got {depth})")


def _check_jobs(self, attribute, n_jobs: int) -> None:
    if n_jobs is not None and n_jobs < -1:
        raise ValueError(
            f"'n_jobs' must be None, or equal to -1, or > 0 (got {n_jobs})"
        )


def _check_location(self, attribute, location: str) -> None:
    if location is not None:
        is_remote = location.startswith("http://") or location.startswith(
            "https://"
        )
        if is_remote and not is_valid_url(location):
            raise ValueError(
                f"'location' must be a valid URL (got {location})"
            )
        elif not is_remote and location is not None:
            if not os.path.exists(location) or not os.path.isfile(location):
                raise FileNotFoundError(
                    f"'location' must be a valid file (got {location})"
                )


def _check_max_walks(self, attribute, max_walks: int) -> None:
    if max_walks is not None and max_walks < 0:
        raise ValueError(f"'max_walks' must be None or > 0 (got {max_walks})")


def is_valid_url(url: str) -> bool:
    """Checks if a URL is valid.

    Args:
        url: The URL to validate.

    Returns:
        True if the URL is valid. False otherwise.

    """
    try:
        requests.get(url)
    except Exception:
        return False
    return True
