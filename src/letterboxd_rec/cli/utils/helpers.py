"""CLI utility helpers for validation, parsing, and notifications."""

import logging
import re

from ...scraper import validate_slug

logger = logging.getLogger(__name__)


def send_notification(message: str) -> None:
    """Send a notification to a configured webhook (Discord/Slack-style)."""
    # Late import to support monkeypatching cli.NOTIFICATION_WEBHOOK_URL in tests
    from letterboxd_rec import cli

    if not cli.NOTIFICATION_WEBHOOK_URL:
        return

    try:
        import httpx

        httpx.post(
            cli.NOTIFICATION_WEBHOOK_URL,
            json={"content": message},
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - best-effort notifications
        logger.warning(f"Failed to send notification: {exc}")


def _require_slug(slug: str) -> str:
    """Validate a film slug and raise if invalid (CLI-friendly wrapper)."""
    cleaned = validate_slug(slug)
    if not cleaned:
        raise ValueError(f"Invalid slug: {slug}")
    return cleaned


# Backward-compat for tests and callers expecting the old name
def _validate_slug(slug: str) -> str:
    return _require_slug(slug)


def _validate_username(username: str) -> str:
    """
    Sanitize a Letterboxd username.
    Returns lowercased alphanumeric + underscores/hyphens only.
    """
    sanitized = re.sub(r'[^a-z0-9_-]', '', username.lower())
    if sanitized != username.lower():
        logger.warning(f"Username '{username}' sanitized to '{sanitized}'")
    return sanitized


def _parse_weights(weights: list[str] | None) -> dict[str, float]:
    """
    Parse CLI weights arguments in the form user:weight into a dict.
    Invalid entries are ignored with a warning.
    """
    if not weights:
        return {}

    parsed: dict[str, float] = {}
    for entry in weights:
        if ":" not in entry:
            logger.warning("Ignoring weight '%s' (expected user:weight)", entry)
            continue
        user_part, weight_part = entry.split(":", 1)
        try:
            weight_value = float(weight_part)
        except ValueError:
            logger.warning("Ignoring weight '%s' (invalid number)", entry)
            continue
        parsed[_validate_username(user_part)] = weight_value

    return parsed
