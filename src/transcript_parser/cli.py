from __future__ import annotations

import argparse


def greet(name: str) -> str:
    """Return a friendly greeting (kept for test compatibility)."""
    return f"Hello, {name}!"


def main(argv: list[str] | None = None) -> None:
    """Minimal CLI used only by tests; unrelated to transcript parsing."""
    parser = argparse.ArgumentParser(description="Transcript_Parser CLI (compat)")
    parser.add_argument("--name", default="world", help="Name to greet")
    args = parser.parse_args(argv)
    print(greet(args.name))


if __name__ == "__main__":
    main()
