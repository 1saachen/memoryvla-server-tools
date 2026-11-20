#!/usr/bin/env python3
"""Extract matching NVIDIA Vulkan userspace libraries locally; never install a system driver."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import urllib.request

DRIVER = '570.148.08'
CHECKSUMS = {
    '22.04': '23e71f0b468be04d68cfbc04128f31ec1fc4c2d74ec813d83d2fb265c115584b',
    '24.04': '0c1ed957fdf9ab04d0a4ca69878efec08dd529151b1fe0be442f5d7602b2c7df',
}


def package_info(os_id, version, driver, arch):
    if os_id != 'ubuntu' or version not in CHECKSUMS or driver != DRIVER or arch != 'x86_64':
        raise ValueError(f'Unverified OS/driver combination: {os_id} {version}, NVIDIA {driver}, {arch}; no changes made')
    base = f'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu{version.replace(".", "")}/x86_64'
    return f'{base}/libnvidia-gl-570_{DRIVER}-1ubuntu1_amd64.deb', CHECKSUMS[version]


def verify_checksum(path, expected):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f'SHA256 mismatch for {path}; extraction refused')


def prepare_environment(prefix, inherited):
    libdir = prefix / 'extracted/usr/lib/x86_64-linux-gnu'
    library = libdir / f'libGLX_nvidia.so.{DRIVER}'
    if not library.is_file():
        raise FileNotFoundError(f'Expected NVIDIA Vulkan driver not found: {library}')
    icd = prefix / 'nvidia_icd.json'
    icd.write_text(json.dumps({'file_format_version': '1.0.0',
                              'ICD': {'library_path': str(library), 'api_version': '1.3.0'}}, indent=2) + '\n')
    env = dict(inherited)
    env.pop('VK_DRIVER_FILES', None)
    env['VK_ICD_FILENAMES'] = str(icd)
    env['LD_LIBRARY_PATH'] = str(libdir) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    lines = ['# Source this file before launching the RoboMME simulation client.',
             'unset VK_DRIVER_FILES', f'export VK_ICD_FILENAMES={shlex.quote(str(icd))}',
             f'export LD_LIBRARY_PATH={shlex.quote(str(libdir))}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}']
    vendor = prefix / 'extracted/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
    if vendor.is_file():
        env['__EGL_VENDOR_LIBRARY_FILENAMES'] = str(vendor)
        lines.append(f'export __EGL_VENDOR_LIBRARY_FILENAMES={shlex.quote(str(vendor))}')
    script = prefix / 'env.sh'
    script.write_text('\n'.join(lines) + '\n')
    return script, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, default=Path('/data/embodied-ai/yaolin/nvidia_vulkan') / DRIVER)
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Run on the Linux evaluation server')
    for tool in ('nvidia-smi', 'dpkg-deb', 'vulkaninfo'):
        if shutil.which(tool) is None:
            raise RuntimeError(f'Required command missing: {tool}')
    release = {}
    for line in Path('/etc/os-release').read_text().splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            release[key] = value.strip('"')
    drivers = set(subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                         text=True, timeout=15).split())
    if len(drivers) != 1:
        raise ValueError(f'Expected one NVIDIA driver version; found {sorted(drivers)}')
    url, checksum = package_info(release.get('ID'), release.get('VERSION_ID'), drivers.pop(), platform.machine())
    prefix = args.prefix.resolve()
    prefix.mkdir(parents=True, exist_ok=True)
    archive = prefix / url.rsplit('/', 1)[1]
    if not archive.exists():
        print(f'Downloading NVIDIA userspace package (~155 MiB): {url}', flush=True)
        partial = archive.with_suffix('.deb.part')
        with urllib.request.urlopen(url, timeout=60) as response, partial.open('wb') as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        verify_checksum(partial, checksum)
        partial.rename(archive)
    verify_checksum(archive, checksum)
    extracted = prefix / 'extracted'
    print(f'Extracting only (no package installation) into {extracted}', flush=True)
    subprocess.run(['dpkg-deb', '-x', str(archive), str(extracted)], check=True)
    script, env = prepare_environment(prefix, os.environ)
    print(f'Environment file ready: source {shlex.quote(str(script))}', flush=True)
    library = json.loads((prefix / 'nvidia_icd.json').read_text())['ICD']['library_path']
    # Use a new process so LD_LIBRARY_PATH is applied before loading NVIDIA libraries.
    subprocess.run([sys.executable, '-c', 'import ctypes,sys; ctypes.CDLL(sys.argv[1]); print("NVIDIA Vulkan library loaded")', library],
                   env=env, check=True)
    result = subprocess.run(['vulkaninfo', '--summary'], env=env, capture_output=True, text=True, timeout=45)
    report = result.stdout + result.stderr
    (prefix / 'vulkan-summary.txt').write_text(report)
    print(report, flush=True)
    if result.returncode or 'NVIDIA H200' not in result.stdout:
        raise RuntimeError(f'H200 Vulkan not confirmed; keep {prefix / "vulkan-summary.txt"} for diagnosis')
    print('VULKAN_H200_OK: NVIDIA H200 is visible to Vulkan. SAPIEN rendering still needs a smoke test.', flush=True)


if __name__ == '__main__':
    main()
