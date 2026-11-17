#!/usr/bin/env python3
"""Download the official Linux simulator with resumable range requests."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import time
import urllib.request
import zipfile

URL = 'http://virtual-home.org/release/simulator/v2.0/v2.3.0/linux_exec.zip'
SIZE = 287682964
CHUNK = 1024 * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('.runtime'))
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    parts = args.output / 'download_parts'
    parts.mkdir(parents=True, exist_ok=True)

    def download(index):
        start, end = index * CHUNK, min(SIZE, (index + 1) * CHUNK) - 1
        path = parts / f'{index:04d}.part'
        if path.exists() and path.stat().st_size == end - start + 1:
            return
        for attempt in range(4):
            try:
                request = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status != 206:
                        raise RuntimeError('Server did not honor the range request')
                    data = response.read()
                if len(data) != end - start + 1:
                    raise RuntimeError('Incomplete range')
                path.write_bytes(data)
                return
            except (OSError, RuntimeError):
                if attempt == 3:
                    raise
                time.sleep(2)

    count = (SIZE + CHUNK - 1) // CHUNK
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done, future in enumerate(as_completed([pool.submit(download, i) for i in range(count)]), 1):
            future.result()
            if done % 10 == 0:
                print(f'{done}/{count} parts', flush=True)
    archive = args.output / 'linux_exec.zip'
    with archive.open('wb') as output:
        for i in range(count):
            output.write((parts / f'{i:04d}.part').read_bytes())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    print(f'SHA256 {digest}')
    destination = (args.output / 'simulator').resolve()
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            if not (destination / name).resolve().is_relative_to(destination):
                raise ValueError('Unsafe archive member')
        if bundle.testzip() is not None:
            raise RuntimeError('Archive CRC check failed')
        bundle.extractall(destination)
    for executable in destination.rglob('*.x86_64'):
        executable.chmod(executable.stat().st_mode | 0o111)
        print(executable)


if __name__ == '__main__':
    main()
