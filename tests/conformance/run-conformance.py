#!/usr/bin/env python

# Copyright 2018-2022 The clvk authors.
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
import pprint
import re
import subprocess
import sys
import tempfile

THIS_DIR = os.path.dirname(__file__)
TOP_DIR = os.path.realpath(os.path.join(THIS_DIR, '..', '..'))
CTS_DIR = os.path.join(TOP_DIR, 'external', 'OpenCL-CTS')

# ('Name', 'binary', 'arg0', 'arg1', ...)
TESTS_QUICK = (
    ('API', 'api/test_api'),
    ('Atomics', 'atomics/test_atomics'),
    ('Compute Info', 'computeinfo/test_computeinfo'),
    ('Common Functions', 'commonfns/test_commonfns'),
    ('Compiler', 'compiler/test_compiler'),
    ('Contractions', 'contractions/test_contractions'),
    ('Device Partitioning', 'device_partition/test_device_partition'),
    ('Device timer', 'device_timer/test_device_timer'),
    ('Events', 'events/test_events'),
    ('Geometric Functions', 'geometrics/test_geometrics'),
    ('Mem (Host Flags)', 'mem_host_flags/test_mem_host_flags'),
    ('Profiling', 'profiling/test_profiling'),
)

TESTS_MODE_WIMPY = (
    ('Conversions', 'conversions/test_conversions', '-w'),
    ('Integer Ops', 'integer_ops/test_integer_ops', 'quick*', 'integer*', 'vector_scalar', 'popcount', 'unary_ops*'),
    ('Half Ops', 'half/test_half', '-w'),
    ('Math', 'math_brute_force/test_bruteforce', '-1', '-w'),
    ('Relationals', 'relationals/test_relationals'),
    ('Select', 'select/test_select', '-w'),
    ('Thread Dimensions', 'thread_dimensions/test_thread_dimensions', 'quick*'),
)

TESTS_MODE_NOT_WIMPY = (
    ('Conversions', 'conversions/test_conversions'),
    ('Integer Ops', 'integer_ops/test_integer_ops'),
    ('Half Ops', 'half/test_half'),
    ('Math', 'math_brute_force/test_bruteforce'),
    ('Relationals', 'relationals/test_relationals'),
    ('Select', 'select/test_select'),
    ('Thread Dimensions', 'thread_dimensions/test_thread_dimensions', 'full*'),
)

TESTS_FOR_WIMPY = TESTS_QUICK + (
    ('Basic', 'basic/test_basic'),
    ('Buffers', 'buffers/test_buffers'),
    ('Vectors', 'vectors/test_vectors'),
    ('C11 Atomics', 'c11_atomics/test_c11_atomics'),
    ('Device execution', 'device_execution/test_device_execution'),
    ('Generic Address Space', 'generic_address_space/test_generic_address_space'),
    ('Half Ops', 'half/test_half'),
    ('Multiple Device/Context', 'multiple_device_context/test_multiples'),
    ('Non-uniform work-group', 'non_uniform_work_group/test_non_uniform_work_group'),
    ('Pipes', 'pipes/test_pipes'),
    ('Printf', 'printf/test_printf'),
    ('SPIR', 'spir/test_spir'),
    ('SPIR-V', 'spirv_new/test_spirv_new', '--spirv-binaries-path', os.path.join(CTS_DIR, 'test_conformance', 'spirv_new', 'spirv_bin')),
    ('SVM', 'SVM/test_svm'),
    ('Subgroups', 'subgroups/test_subgroups'),
    ('Workgroups', 'workgroups/test_workgroups'),
)

TESTS_IMAGES_FAST = (
    ('Images (API Info)', 'images/clGetInfo/test_cl_get_info'),
    ('Images (clReadWriteImage)', 'images/clReadWriteImage/test_cl_read_write_images'),
    ('Images (clReadWriteImage pitch)', 'images/clReadWriteImage/test_cl_read_write_images', 'use_pitches'),
    ('Images (clReadWriteImage max size)', 'images/clReadWriteImage/test_cl_read_write_images', 'max_images'),
    ('Images (clFillImage)', 'images/clFillImage/test_cl_fill_images'),
    ('Images (clFillImage pitch)', 'images/clFillImage/test_cl_fill_images', 'use_pitches'),
    ('Images (clFillImage max size)', 'images/clFillImage/test_cl_fill_images', 'max_images'),
    ('Images (Samplerless)', 'images/samplerlessReads/test_samplerless_reads'),
    ('Images (Samplerless pitch)', 'images/samplerlessReads/test_samplerless_reads', 'use_pitches'),
    ('Images (Samplerless max size)', 'images/samplerlessReads/test_samplerless_reads', 'max_images'),
    ('Images (Kernel Methods)', 'images/kernel_image_methods/test_kernel_image_methods'),
)


TESTS_IMAGES_SLOW = (
    ('Images (Kernel)', 'images/kernel_read_write/test_image_streams', 'CL_FILTER_NEAREST'),
    ('Images (Kernel pitch)', 'images/kernel_read_write/test_image_streams', 'use_pitches', 'CL_FILTER_NEAREST'),
    ('Images (Kernel max size)', 'images/kernel_read_write/test_image_streams', 'max_images', 'CL_FILTER_NEAREST'),
    ('Images (clCopyImage)', 'images/clCopyImage/test_cl_copy_images'),
    ('Images (clCopyImage small)', 'images/clCopyImage/test_cl_copy_images', 'small_images'),
    ('Images (clCopyImage max size)', 'images/clCopyImage/test_cl_copy_images', 'max_images'),
)

TESTS_EXTENSIONS = (
    ('cl_ext_cxx_for_opencl', 'test_cl_ext_cxx_for_opencl'),
    ('cl_khr_command_buffer', 'test_cl_khr_command_buffer'),
    ('cl_khr_semaphore', 'test_cl_khr_semaphore'),
    ('External memory and synchronisation', 'test_vulkan'),
)

TESTS_IMAGES = TESTS_IMAGES_FAST + TESTS_IMAGES_SLOW

TESTS_WIMPY = TESTS_FOR_WIMPY + TESTS_MODE_WIMPY + TESTS_IMAGES_FAST + TESTS_EXTENSIONS

TESTS_FULL_CONFORMANCE = TESTS_FOR_WIMPY + TESTS_MODE_NOT_WIMPY + TESTS_IMAGES + TESTS_EXTENSIONS + (
    ('Allocations (single maximum)', 'allocations/test_allocations', 'single', '5', 'all'),
    ('Allocations (total maximum)', 'allocations/test_allocations', 'multiple', '5', 'all'),
#    ('CL_DEVICE_TYPE_CPU, Images (Kernel CL_FILTER_LINEAR),images/kernel_read_write/test_image_streams CL_FILTER_LINEAR
#    ('CL_DEVICE_TYPE_CPU, Images (Kernel CL_FILTER_LINEAR pitch),images/kernel_read_write/test_image_streams use_pitches CL_FILTER_LINEAR
#    ('CL_DEVICE_TYPE_CPU, Images (Kernel CL_FILTER_LINEAR max size),images/kernel_read_write/test_image_streams max_images CL_FILTER_LINEAR
#    ('OpenCL-GL Sharing,gl/test_gl
)

TEST_SETS = {
    'quick': TESTS_QUICK,
    'wimpy': TESTS_WIMPY,
    'full': TESTS_FULL_CONFORMANCE,
    'images': TESTS_IMAGES,
    'images-fast': TESTS_IMAGES_FAST,
    'images-slow': TESTS_IMAGES_SLOW,
    'extensions': TESTS_EXTENSIONS,
}

TIME_SERIALISATION_FORMAT = '%H:%M:%S.%f'

COLOUR_RED = '\033[0;31m'
COLOUR_GREEN = '\033[0;32m'
COLOUR_YELLOW = '\033[0;33m'
COLOUR_RESET = '\033[0m'

def timedelta_from_string(string):
    duration_as_date = datetime.datetime.strptime(string, TIME_SERIALISATION_FORMAT)
    return duration_as_date - datetime.datetime(1900, 1, 1)

def timedelta_to_string(duration):
    d = duration
    duration_as_date = datetime.datetime(
        year=1900, month=1, day=d.days+1,
        hour=int(d.seconds / 3600),
        minute=int((d.seconds / 60) % 60),
        second=int(d.seconds % 60),
        microsecond=d.microseconds
    )
    return datetime.datetime.strftime(duration_as_date, TIME_SERIALISATION_FORMAT)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def run_conformance_binary(path, testargs, args):
    start = datetime.datetime.utcnow()
    dirname = os.path.dirname(path)
    binary = os.path.basename(path)
    path = os.path.join(dirname, os.path.basename(binary))
    workdir = os.path.dirname(path)
    result_json = os.path.join(workdir, 'conf.json')
    os.environ['CL_CONFORMANCE_RESULTS_FILENAME'] = result_json
    if args.vulkan_loader_json_path:
        print("Using ICD JSON: " + args.vulkan_loader_json_path)
        os.environ['VK_ICD_FILENAMES'] = args.vulkan_loader_json_path
    if os.name == 'nt':
        path += '.exe'
    print(path)
    os.environ['VK_LOADER_DEBUG'] = 'all'
    import pprint
    pprint.pprint(os.environ)
    os.environ['LD_LIBRARY_PATH'] = args.opencl_lib_dir
    os.environ['DYLD_LIBRARY_PATH'] = args.opencl_lib_dir
    p = subprocess.Popen(
        [path] + testargs,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=workdir
    )
    stdout, stderr = p.communicate()
    stdout = stdout.decode('utf-8')
    stderr = stderr.decode('utf-8')
    end = datetime.datetime.utcnow()
    print(stderr)
    print(stdout)
    duration = end - start

    has_results = False
    results = {}

    try:
        data = load_json(result_json)
        os.remove(result_json)

        for test, result in data['results'].items():
            if result not in ('pass', 'skip', 'fail'):
                raise Exception("Can't parse results")
        results = data['results']
        has_results = True
    except Exception as e:
        pass

    return {
        'has_results': has_results,
        'results': results,
        'retcode': p.returncode,
        'duration': timedelta_to_string(duration),
    }

def get_suite_totals(suite_results):
    totals = {
        'pass': 0,
        'fail': 0,
        'skip': 0,
        'total': 0
    }
    for test, result in suite_results['results'].items():
        totals[result] += 1
    totals['total'] = totals['pass'] + totals['fail']
    return totals

def gather_system_info():
    tmpjson = tempfile.mkstemp()[1]
    cmd = [ 'vulkaninfo', '-o', tmpjson, '--json=0' ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise e
    info_raw = load_json(tmpjson)
    info_props = info_raw['capabilities']['device']['properties']
    infos = {
        'deviceID': info_props['VkPhysicalDeviceProperties']['deviceID'],
        'deviceName': info_props['VkPhysicalDeviceProperties']['deviceName'],
        'deviceType': info_props['VkPhysicalDeviceProperties']['deviceType'],
        'driverVersion': info_props['VkPhysicalDeviceProperties']['driverVersion'],
        'apiVersion': info_props['VkPhysicalDeviceProperties']['apiVersion'],
    }
    if 'VkPhysicalDeviceDriverProperties' in info_props:
        infos['driverID'] = info_props['VkPhysicalDeviceDriverProperties']['driverID']
        infos['driverName'] = info_props['VkPhysicalDeviceDriverProperties']['driverName']
        infos['driverInfo'] = info_props['VkPhysicalDeviceDriverProperties']['driverInfo']
    print("System information:")
    pprint.pprint(infos, indent=2)
    print("")
    return infos

def check_system_info(system_info, reference_results_path):
    ref = load_json(reference_results_path)
    refsys = ref['system-info']
    sys_info_checks = ('deviceID','deviceName','deviceType','driverID','driverName')
    success = True
    for sic in sys_info_checks:
        if refsys[sic] != system_info[sic]:
            print("reference for '{}' is '{}' but got '{}'".format(sic, refsys[sic],system_info[sic]))
            success = False
    return success

def run_tests(args):
    test_set = TEST_SETS[args.test_set]
    results = {}

    for test in test_set:
        name = test[0]
        binary = test[1]
        test_args = test[2:]
        if args.filter and not re.match(args.filter, name):
            continue
        print("Running", name, "...")
        abspath = os.path.join(args.cts_install_dir, os.path.basename(binary))
        status = run_conformance_binary(abspath, list(test_args), args)
        results[name] = status
        totals = get_suite_totals(status)
        print("Done, retcode = %d [%s]." % (status['retcode'], status['duration']))
        print(totals['pass'], "test(s) out of", totals['total'], "passed. ", totals['skip'], "were skipped.")
        print("")

    return results

def check_reference(results, reference, args):
    print("")
    print("Difference w.r.t. reference results:")
    has_diff = False
    for name in sorted(results):
        # Skip checking if there is no reference
        if not name in reference:
            print("No reference for {}".format(name))
            continue

        res = results[name]
        ref = reference[name]
        msgs = []

        # Compare retcodes
        if res['retcode'] != ref['retcode']:
            msgs.append("Expected the return code to be {} but got {}".format(ref['retcode'], res['retcode']))

        # Work out test status differences
        for test, result in res['results'].items():

            if test not in ref['results']:
                msgs.append("No reference for test '{}'".format(test))
                continue

            expected_result = ref['results'][test]

            if result != expected_result:
                msgs.append("Expected the status of '{}' to be '{}' but got '{}'".format(test, expected_result, result))

        for test in ref['results']:
            if test not in res['results']:
                msgs.append("Got reference for test '{}' but no result".format(test))

        # Calculate time difference
        if args.compare_duration:
            refdur = timedelta_from_string(ref['duration']).total_seconds()
            resdur = timedelta_from_string(res['duration']).total_seconds()
            timediff = resdur - refdur
            reltimediff = timediff / refdur
            duration_threshold = 0.10
            if abs(reltimediff) > duration_threshold:
                msgs.append('duration, expected {} (+/- {}%) but got {} ({}%)'.format(refdur, duration_threshold * 100, resdur, reltimediff * 100))

        # Print differences if any
        if msgs:
            has_diff = True
            print("\t{}".format(name))
            for msg in msgs:
                print("\t\t{}".format(msg))
    if has_diff:
        return False
    else:
        print("\tNone.")
        return True

def report(results, args):
    total = 0
    passed = 0
    skipped = 0
    bin_success = 0
    with_results = 0
    time = datetime.timedelta()

    tests_passed_template = '{:3d}/{:3d} tests passed, {:3d} skipped'
    line_template = '{name:{name-len}} {result_str:>{result_str-len}} [{duration}]'
    binary_summary_template = '{:2d}/{:2d} binaries produced results'

    # Work out the length of the fields
    len_binaries_summary = len(binary_summary_template.format(0,0))
    len_longest_name = max([len(n) for n in results]+[len_binaries_summary])
    len_result_str = len(tests_passed_template.format(0,0,0))

    # Print results
    all_suites_passing = True
    for name in sorted(results):
        status = results[name]
        suite_totals = get_suite_totals(status)
        has_results = status['has_results']
        # Print status
        if not has_results:
            result_str = 'NO RESULTS'
        else:
            with_results += 1
            result_str = tests_passed_template.format(suite_totals['pass'], suite_totals['total'], suite_totals['skip'])

        line = line_template.format(**{
            'name': name,
            'name-len': len_longest_name,
            'result_str': result_str,
            'result_str-len': len_result_str,
            'duration': status['duration'],
        })

        all_tests_passing = (suite_totals['pass'] == suite_totals['total']) and has_results
        if not all_tests_passing:
            all_suites_passing = False

        colour = COLOUR_GREEN if all_tests_passing else COLOUR_RED
        print(colour + line + COLOUR_RESET)

        # Accumulate totals
        total += suite_totals['total']
        passed += suite_totals['pass']
        skipped += suite_totals['skip']
        time += timedelta_from_string(status['duration'])
        if status['retcode'] == 0:
            bin_success += 1

    line = line_template.format(**{
        'name': binary_summary_template.format(with_results, len(results)),
        'name-len': len_longest_name,
        'result_str': tests_passed_template.format(passed, total, skipped),
        'result_str-len': len_result_str,
        'duration': time,
    })
    colour = COLOUR_GREEN if all_suites_passing else COLOUR_RED
    print('-' * len(line))
    print(colour + line + COLOUR_RESET)
    print("")

    success = True
    if args.reference_results:
        reference = load_json(args.reference_results)
        success = check_reference(results, reference['test-results'], args)
    return success

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

    parser.add_argument(
        '--compare-duration', action='store_true',
        help="Compare tests' execution time with reference values",
    )

    parser.add_argument(
        '--compare-only', action='store_true',
        help="Only compare results to reference",
    )
    parser.add_argument(
        '--vulkan-loader-json-path', default=None,
        help='Path to Vulkan Loader ICD JSON',
    )

    parser.add_argument(
        '--cts-install-dir',
        default=os.path.join(TOP_DIR, 'build', 'conformance'),
        help='Path to installed CTS',
    )
    
    parser.add_argument(
        '--opencl-lib-dir',
        default=os.path.join(TOP_DIR, 'build'),
        help='Path to OpenCL library',
    )

    parser.add_argument(
        '--filter',
        help="Only run tests that match this regexp",
    )

    args = parser.parse_args()

    # Run tests or load results
    if not args.compare_only:
        system_info = gather_system_info()
        if args.reference_results and not check_system_info(system_info, args.reference_results):
            print("\nLooks like the reference results do not correspond to the device being targeting, aborting run.\n")
            sys.exit(1)
        test_results = run_tests(args)
        results = {
            'system-info': system_info,
            'test-results': test_results
        }
        if args.save_results:
            with open(args.save_results, 'w') as f:
                json.dump(results, f, indent=2, sort_keys=True, separators=(',', ': '))
    else:
        results = load_json(args.save_results)

    # Process results
    if not report(results['test-results'], args):
        sys.exit(1)

if __name__ == '__main__':
    main()
