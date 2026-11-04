"""Select an existing reconstruction/metric policy without changing its CLI flags."""

import argparse
import importlib
import sys


POLICIES = {
    "solved_only": "connections_research.reconstruction.policies.solved_only",
    "previous_guess": "connections_research.reconstruction.policies.previous_guess",
    "a_plus_h_v1": "connections_research.reconstruction.policies.a_plus_h_v1",
    "a_plus_h_v2": "connections_research.reconstruction.policies.a_plus_h_v2",
}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Compute trajectory metrics using an explicit reconstruction policy.",
        epilog="Use --policy NAME --help to see that policy's original arguments.",
        add_help=False,
    )
    parser.add_argument("--policy", choices=sorted(POLICIES), required=True)
    # Forward help to the selected policy, but show selector help on its own.
    if not arguments or ("--policy" not in arguments and not any(a.startswith("--policy=") for a in arguments)
                         and any(a in {"-h", "--help"} for a in arguments)):
        parser.print_help()
        return 0
    selected, remaining = parser.parse_known_args(arguments)
    policy = importlib.import_module(POLICIES[selected.policy])
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        policy.main()
    finally:
        sys.argv = original_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
