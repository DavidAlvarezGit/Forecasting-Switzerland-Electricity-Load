from typing import List


class ValidationError(Exception):
    pass


def collect_checks(checks: List):
    """
    Run a list of callables and collect all AssertionErrors.
    Raises a single ValidationError with all messages.
    """
    errors = []

    for check in checks:
        try:
            check()
        except AssertionError as e:
            errors.append(str(e))

    if errors:
        raise ValidationError("\n".join(errors))