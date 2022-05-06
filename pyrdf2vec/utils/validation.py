import os

import attr
import requests


def _check_max_depth(self, attribute: attr.Attribute, depth: int) -> None:
    """Checks if a given max depth is valid for a walking strategy.

    Args:
        attribute: The attribute.
        max_depth: The maximum depth of the walk to check the validity.

    Raises:
        ValueError: If the maximum depth is invalid.

    """
    if depth < 0:
        raise ValueError(f"'depth' must be >= 0 (got {depth})")


def _check_jobs(self, attribute: attr.Attribute, n_jobs: int) -> None:
    """Checks if a given number of processes is correct.

    Args:
        attribute: The attribute.
        n_jobs: The number of processes to check the validity.

    Raises:
        ValueError: If the number of processes is invalid.

    """
    if n_jobs is not None and n_jobs < -1:
        raise ValueError(
            f"'n_jobs' must be None, or equal to -1, or > 0 (got {n_jobs})"
        )


def _check_location(self, attribute: attr.Attribute, location: str) -> None:
    """Checks if a given file can be accessed locally or remotely.

    Args:
        attribute: The attribute.
        location: The file location or URL to check the validity.

    Raises:
        FileNotFoundError: If the file should be accessible locally but the
            location is invalid.
        ValueError: If the file should be accessible remotely but the URL is
            invalid.

    """
    if location is not None:
        is_remote = location.startswith("http://") or location.startswith(
            "https://"
        )
        if is_remote and not is_valid_url(location):
            raise ValueError(
                f"'location' must be a valid URL (got {location})"
            )
        elif not is_remote:
            if not os.path.exists(location) or not os.path.isfile(location):
                raise FileNotFoundError(
                    f"'location' must be a valid file (got {location})"
                )


def _check_max_walks(self, attribute: attr.Attribute, max_walks: int) -> None:
    """Checks if a given number of maximum walks per entity is correct.

    Args:
        attribute: The attribute.
        max_walks: The maximum walks per entity to check the validity.

    Raises:
        ValueError: If the maximum walks per entity is invalid.

    """
    if max_walks is not None and max_walks < 0:
        raise ValueError(f"'max_walks' must be None or > 0 (got {max_walks})")


def is_valid_url(url: str) -> bool:
    """Checks if a given URL is valid.

    Args:
        url: The URL to check the validity.

    Returns:
        True if the URL is valid, False otherwise.

    """
    try:
        query = "ASK {}"
        res_code = requests.head(url, params={"query": query}).status_code
        return res_code == requests.codes.ok
    except Exception:
        return False
