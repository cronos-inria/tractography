import argparse
import pkgutil

import tractography as tg
from . import utils


DESCRIPTION = """\
"""


def parse_arguments():

    parser = argparse.ArgumentParser(description=DESCRIPTION)
    subparsers = parser.add_subparsers()
    subparsers.required = True
    subparsers.dest = "subcommand"

    # Load all the sub commands from the streamlines.cli.commands package
    # dynamically.
    package = tg.cli
    prefix = package.__name__ + "."
    for _, name, _ in pkgutil.iter_modules(package.__path__, prefix):
        module = __import__(name, fromlist=["nothing"])
        if "add_parser" in module.__dict__:
            module.add_parser(subparsers)

    return parser.parse_args()


def main():

    args = parse_arguments()
    parameters = {k: v for k, v in vars(args).items() if k not in ["func", "subcommand"]}
    args.func(**parameters)


if __name__ == "__main__":
    main()
