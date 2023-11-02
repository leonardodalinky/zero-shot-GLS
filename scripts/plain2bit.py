"""Plain-text to bitstring.

Usage:
    TODO
"""
import argparse
import os.path as osp
import sys

sys.path.append(f"{osp.dirname(osp.abspath(__file__))}/../src")

parser = argparse.ArgumentParser(description="Plain-text to bitstring.")
parser.add_argument("input", type=str, help="Input .csv file.")
parser.add_argument(
    "-s",
    "--src-col",
    type=str,
    default="plaintext",
    help="Column name of source plain-text. [Default: plaintext]",
)
parser.add_argument(
    "-d",
    "--dst-col",
    type=str,
    default="bits",
    help="Column name of destination bitstring. [Default: enc_bits]",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    required=True,
    help="Output .csv file.",
)

if __name__ == "__main__":
    ...
