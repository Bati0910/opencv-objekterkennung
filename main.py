import sys


def main() -> int:
    try:
        from objekterkennung.ui.app import main as run_interface
        return run_interface()
    except (OSError, ValueError, ImportError) as error:
        print(f"Start nicht möglich: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
