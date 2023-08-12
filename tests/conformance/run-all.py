#!/usr/bin/env python

import argparse
import os
import subprocess

THIS_DIR = os.path.dirname(__file__)
TOP_DIR = os.path.realpath(os.path.join(THIS_DIR, '..', '..'))
DEFAULT_BUILD_DIR = os.path.join(TOP_DIR, 'build')
TESTS_CONF_DIR = os.path.join(TOP_DIR, 'tests', 'conformance')

CONFIGS = (
    {
        'env': {
            'MESA_VK_DEVICE_SELECT': '8086:56a1',
        },
        'results': 'results-intel-a750.json',
    },
    {
        'env': {
            'MESA_VK_DEVICE_SELECT': '1002:164e',
        },
        'results': 'results-amd-7950x.json',
    },
    {
        'env': {
            'MESA_VK_DEVICE_SELECT': '10005:0',
        },
        'results': 'results-llvmpipe.json',
    },
    {
        'env': {
            'MESA_VK_DEVICE_SELECT': '8086:56a1',
            'CLVK_PHYSICAL_ADDRESSING': '1',
            'CLVK_SPIRV_ARCH': 'spir64',
        },
        'results': 'results-intel-a750-physical.json',
    },
    {
        'env': {
            'MESA_VK_DEVICE_SELECT': '1002:164e',
            'CLVK_PHYSICAL_ADDRESSING': '1',
            'CLVK_SPIRV_ARCH': 'spir64',
        },
        'results': 'results-amd-7950x-physical.json',
    },
    {
        'env': {
            'MESA_VK_DEVICE_SELECT': '10005:0',
            'CLVK_PHYSICAL_ADDRESSING': '1',
            'CLVK_SPIRV_ARCH': 'spir64',
        },
        'results': 'results-llvmpipe-physical.json',
    },

)

parser = argparse.ArgumentParser()

parser.add_argument('--build-dir', default=DEFAULT_BUILD_DIR, help="Path to build directory")

args = parser.parse_args()

for config in CONFIGS:
    env = dict(config['env'])
    results = config['results']
    ref = os.path.join(TESTS_CONF_DIR, results)
    runner_script = os.path.join(TESTS_CONF_DIR, 'run-conformance.py')
    cmd = [runner_script, '--jobs', '8', '--save-results', results]
    if os.path.isfile(ref):
        cmd += ['--reference-results', ref]
    env['LD_LIBRARY_PATH'] = os.path.join(args.build_dir)
    subprocess.run(cmd, check=True, env=env)
