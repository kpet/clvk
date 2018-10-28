#!/usr/bin/env python

# Copyright 2018 The clvk authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import datetime
import json
import os
import re
import subprocess

THIS_DIR = os.path.dirname(__file__)
TOP_DIR = os.path.realpath(os.path.join(THIS_DIR, '..', '..'))
CONFORMANCE_DIR = os.path.join(TOP_DIR, 'build', 'external', 'OpenCL-CTS', 'test_conformance')

# ('Name', 'binary', 'arg0', 'arg1', ...)
TESTS_HEADERS = (
    ('Headers (cl_typen)', 'headers/test_headers'),
    ('Headers (cl.h standalone)', 'headers/test_cl_h'),
    ('Headers (cl_platform.h standalone)', 'headers/test_cl_platform_h'),
    ('Headers (cl_gl.h standalone)', 'headers/test_cl_gl_h'),
    ('Headers (opencl.h standalone)', 'headers/test_opencl_h'),
)

TESTS_QUICK = TESTS_HEADERS + (
    ('API', 'api/test_api'),
    ('Atomics', 'atomics/test_atomics'),
    ('Compute Info', 'computeinfo/computeinfo'),
    ('Common Functions', 'commonfns/test_commonfns'),
    ('Compiler', 'compiler/test_compiler'),
    ('Contractions', 'contractions/contractions'),
    ('Device Partitioning', 'device_partition/test_device_partition'),
    ('Events', 'events/test_events'),
    ('Geometric Functions', 'geometrics/test_geometrics'),
    ('Half Ops', 'half/Test_half'),
    ('Mem (Host Flags)', 'mem_host_flags/test_mem_host_flags'),
    ('Multiple Device/Context', 'multiple_device_context/test_multiples'),
    ('Printf', 'printf/test_printf'),
    ('Profiling', 'profiling/test_profiling'),
    ('VecAlign', 'vec_align/test_vecalign'),
    ('VecStep', 'vec_step/test_vecstep'),
)

TESTS_MODE_WIMPY = (
    ('Math', 'math_brute_force/bruteforce', '-1', '-w'),
    ('Relationals', 'relationals/test_relationals', 'relational_*'),
    ('Select', 'select/test_select', '-w'),
    ('Thread Dimensions', 'thread_dimensions/test_thread_dimensions', 'quick*'),
)

TESTS_MODE_NOT_WIMPY = (
    ('Math', 'math_brute_force/bruteforce'),
    ('Conversions', 'conversions/test_conversions'),
    ('Integer Ops', 'integer_ops/test_integer_ops'),
    ('Relationals', 'relationals/test_relationals'),
    ('Select', 'select/test_select'),
    ('Thread Dimensions', 'thread_dimensions/test_thread_dimensions', 'full*'),
)

TESTS_FOR_WIMPY = TESTS_QUICK + (
    ('Basic', 'basic/test_basic'),
    ('Buffers', 'buffers/test_buffers'),
)

TESTS_WIMPY = TESTS_FOR_WIMPY + TESTS_MODE_WIMPY

TESTS_IMAGES = (
    ('Images (API Info)', 'images/clGetInfo/test_cl_get_info'),
    ('Images (Kernel Methods)', 'images/kernel_image_methods/test_kernel_image_methods'),
    ('Images (Kernel)', 'images/kernel_read_write/test_image_streams', 'CL_FILTER_NEAREST'),
    ('Images (Kernel pitch)', 'images/kernel_read_write/test_image_streams', 'use_pitches', 'CL_FILTER_NEAREST'),    ('Images (Kernel max size)', 'images/kernel_read_write/test_image_streams', 'max_images', 'CL_FILTER_NEAREST'),
    ('Images (clCopyImage)', 'images/clCopyImage/test_cl_copy_images'),
    ('Images (clCopyImage small)', 'images/clCopyImage/test_cl_copy_images', 'small_images'),
    ('Images (clCopyImage max size)', 'images/clCopyImage/test_cl_copy_images', 'max_images'),
    ('Images (clReadWriteImage)', 'images/clReadWriteImage/test_cl_read_write_images'),
    ('Images (clReadWriteImage pitch)', 'images/clReadWriteImage/test_cl_read_write_images', 'use_pitches'),
    ('Images (clReadWriteImage max size)', 'images/clReadWriteImage/test_cl_read_write_images', 'max_images'),
    ('Images (clFillImage)', 'images/clFillImage/test_cl_fill_images'),
    ('Images (clFillImage pitch)', 'images/clFillImage/test_cl_fill_images', 'use_pitches'),
    ('Images (clFillImage max size)', 'images/clFillImage/test_cl_fill_images', 'max_images'),
    ('Images (Samplerless)', 'images/samplerlessReads/test_samplerless_reads'),
    ('Images (Samplerless pitch)', 'images/samplerlessReads/test_samplerless_reads', 'use_pitches'),
    ('Images (Samplerless max size)', 'images/samplerlessReads/test_samplerless_reads', 'max_images'),
)



TESTS_FULL_CONFORMANCE = TESTS_FOR_WIMPY + TESTS_MODE_NOT_WIMPY + TESTS_IMAGES + (
    ('Allocations (single maximum)', 'allocations/test_allocations', 'single', '5', 'all'),
    ('Allocations (total maximum)', 'allocations/test_allocations', 'multiple', '5', 'all'),
#    ('Headers (cl.h standalone C99), headers/test_cl_h_c99
#    ('Headers (cl_platform.h standalone C99), headers/test_cl_platform_h_c99
#    ('Headers (cl_gl.h standalone C99), headers/test_cl_gl_h_c99
#    ('Headers (opencl.h standalone C99), headers/test_opencl_h_c99
#    ('CL_DEVICE_TYPE_CPU, Images (Kernel CL_FILTER_LINEAR),images/kernel_read_write/test_image_streams CL_FILTER_LINEAR
#    ('CL_DEVICE_TYPE_CPU, Images (Kernel CL_FILTER_LINEAR pitch),images/kernel_read_write/test_image_streams use_pitches CL_FILTER_LINEAR
#    ('CL_DEVICE_TYPE_CPU, Images (Kernel CL_FILTER_LINEAR max size),images/kernel_read_write/test_image_streams max_images CL_FILTER_LINEAR
#    ('OpenCL-GL Sharing,gl/test_gl
)

TEST_SETS = {
    'quick': TESTS_QUICK,
    'wimpy': TESTS_WIMPY,
    'full': TESTS_FULL_CONFORMANCE,
}

TIME_SERIALISATION_FORMAT = '%H:%M:%S.%f'

def timedelta_from_string(string):
    duration_as_date = datetime.datetime.strptime(string, TIME_SERIALISATION_FORMAT)
    return duration_as_date - datetime.datetime(1900, 1, 1)

def timedelta_to_string(duration):
    d = duration
    duration_as_date = datetime.datetime(
        year=1900, month=1, day=d.days+1,
        hour=d.seconds / 3600,
        minute=d.seconds / 60,
        second=d.seconds % 60,
        microsecond=d.microseconds
    )
    return datetime.datetime.strftime(duration_as_date, TIME_SERIALISATION_FORMAT)

def run_conformance_binary(path, args):
    start = datetime.datetime.utcnow()
    dirname = os.path.dirname(path)
    binary = os.path.basename(path)
    path = os.path.join(dirname, 'conformance_' + binary)
    p = subprocess.Popen(
        [path] + args,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=os.path.dirname(path)
    )
    stdout, stderr = p.communicate()
    end = datetime.datetime.utcnow()
    #print(stdout)
    duration = end - start

    tests_info_found = False

    if not tests_info_found:
        tinfo = re.findall(r'PASSED (\d+) of (\d+) tests.',stdout)
        if tinfo:
            tests_total = int(tinfo[0][1])
            tests_passed = int(tinfo[0][0])
            tests_info_found = True

    if not tests_info_found:
        tinfo = re.findall(r'FAILED (\d+) of (\d+) tests.',stdout)
        if tinfo:
            tests_total = int(tinfo[0][1])
            tests_passed = tests_total - int(tinfo[0][0])
            tests_info_found = True

    if not tests_info_found:
        if re.search(r'PASSED \S+.', stdout):
            tests_passed = 1
            tests_total = 1
            tests_info_found = True

    if not tests_info_found:
        if re.search(r'FAILED \S+.', stdout):
            tests_passed = 0
            tests_total = 1
            tests_info_found = True

    if not tests_info_found:
        tests_total = 0
        tests_passed = 0

    return {
        'total': tests_total,
        'passed': tests_passed,
        'retcode': p.returncode,
        'duration': timedelta_to_string(duration),
    }

def run_tests(test_set):

    results = {}

    for test in test_set:
        name = test[0]
        binary = test[1]
        args = test[2:]
        print("Running", name, "...")
        status = run_conformance_binary(os.path.join(CONFORMANCE_DIR, binary), list(args))
        results[name] = status
        print("Done, retcode = %d [%s]." % (status['retcode'], status['duration']))
        print(status['passed'], "test(s) out of", status['total'], "passed")
        print("")

    return results

def check_reference(results, reference):
    print("")
    print("Difference w.r.t. reference results:")
    for name in sorted(results):
        # Skip checking if there is no reference
        if not reference.has_key(name):
            print("No reference for {}".format(name))
            continue

        res = results[name]
        ref = reference[name]
        keys = ('total', 'passed', 'retcode')
        differences = []
        time_msg = None

        # Calculate differences on numeric keys
        for k in keys:
            if res[k] != ref[k]:
                differences.append((k, ref[k], res[k]))

        # Calculate time difference
        refdur = timedelta_from_string(ref['duration']).total_seconds()
        resdur = timedelta_from_string(res['duration']).total_seconds()
        timediff = resdur - refdur
        reltimediff = timediff / refdur
        duration_threshold = 0.05
        if abs(reltimediff) > duration_threshold:
            time_msg = '\t\tduration, expected {} (+/- {}%) but got {} ({}%)'.format(refdur, duration_threshold * 100, resdur, reltimediff * 100)
        # Print differences if any
        if differences or time_msg:
            print("\t{}".format(name))
            for d in differences:
                print("\t\t{}, expected {} but got {} ({:+d})".format(d[0], d[1], d[2], d[2] - d[1]))
            if time_msg:
                print(time_msg)

def report(results, args):

    total = 0
    passed = 0
    bin_success = 0
    with_results = 0
    time = datetime.timedelta()

    tests_passed_template = '{:3d}/{:3d} tests passed'
    line_template = '{name:{name-len}} {result_str:>{result_str-len}} [{duration}]'
    binary_summary_template = '{:2d}/{:2d} binaries produced results'

    # Work out the length of the fields
    len_binaries_summary = len(binary_summary_template.format(0,0))
    len_longest_name = max([len(n) for n in results]+[len_binaries_summary])
    len_result_str = len(tests_passed_template.format(0,0))

    # Print results
    for name in sorted(results):
        status = results[name]
        # Print status
        if status['total'] == 0:
            result_str = 'NO RESULTS'
        else:
            with_results += 1
            result_str = tests_passed_template.format(status['passed'], status['total'])

        line = line_template.format(**{
            'name': name,
            'name-len': len_longest_name,
            'result_str': result_str,
            'result_str-len': len_result_str,
            'duration': status['duration'],
        })

        print(line)

        # Accumulate totals
        total += status['total']
        passed += status['passed']
        time += timedelta_from_string(status['duration'])
        if status['retcode'] == 0:
            bin_success += 1

    line = line_template.format(**{
        'name': binary_summary_template.format(with_results, len(results)),
        'name-len': len_longest_name,
        'result_str': tests_passed_template.format(passed, total),
        'result_str-len': len_result_str,
        'duration': time,
    })
    print('-' * len(line))
    print(line)
    print("")

    if args.reference_results:
        with open(args.reference_results) as f:
            reference = json.load(f)
        check_reference(results, reference)

def main():

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--save-results',
        help='Save results to a JSON file',
    )

    parser.add_argument(
        '--reference-results',
        help='Check results against JSON reference',
    )

    parser.add_argument(
        '--test-set', choices=TEST_SETS.keys(), default='wimpy',
        help='The set of tests to run',
    )

    args = parser.parse_args()

    # Run tests
    results = run_tests(TEST_SETS[args.test_set])
    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2, sort_keys=True, separators=(',', ': '))

    # Process results
    report(results, args)

if __name__ == '__main__':
    main()
