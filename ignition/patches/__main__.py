"""Apply Inductor fixes in the Python environment selected by uv run."""

import sys

from ignition.errors import PreparationError
from . import apply


def main() -> int:
    try:
        for path in apply():
            print(f"Patched {path}")
        return 0
    except PreparationError as error:
        print(f"error: {error}", file=sys.stderr)
        return error.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
