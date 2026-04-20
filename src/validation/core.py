
from typing import Callable


class ValidationError(Exception):
    pass


def collect_checks(checks: list[Callable[[], None]]) -> None:
    """
    Run a list of callables and collect all AssertionErrors.
    Raises a single ValidationError with all messages.
    """
    errors: list[str] = []

    for check in checks:
        try:
            check()
        except AssertionError as e:
            errors.append(str(e))

    if errors:
        raise ValidationError("\n".join(errors))