"""Combine complete methods or disjoint shards under one frozen split."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from sgmem.shards import merge_runs
from sgmem.retrieval import METHODS


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts', nargs='+', required=True, help='Completed method directories')
    parser.add_argument('--split', required=True, help='Original full dev/test split')
    parser.add_argument('--output', required=True, help='New aggregate directory')
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=['bm25', 'dense'])
    parser.add_argument('--partition', choices=('dev', 'test', 'all'), default='all')
    args = parser.parse_args()
    summary = merge_runs(args.parts, args.split, args.output, ROOT, methods=args.methods, partition=args.partition)
    for method, metrics in summary.items():
        print(method, metrics['overall'])
