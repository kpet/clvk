#!/usr/bin/env python

import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--input', required=True)
parser.add_argument('--runner-os', required=True)
args = parser.parse_args()

PREFIXES = {
    'Linux': './',
    'macOS': './',
    'Windows': '.\\',
}

with open(args.input) as f:
    data = json.load(f)
    lpath = data['ICD']['library_path']
    print(f"Origial library_path: {lpath}")
    lpath = PREFIXES[args.runner_os] + os.path.basename(lpath)
    print(f"Fixed up library_path: {lpath}")
    data['ICD']['library_path'] = lpath

with open(args.output, 'w') as f:
    json.dump(data, f)
