"""Link or copy original videos and verify their capture hashes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    sources=parser.add_mutually_exclusive_group()
    sources.add_argument('--source',type=Path)
    sources.add_argument('--source-dataset',type=Path)
    args=parser.parse_args()
    manifest=json.loads((args.dataset/'video_manifest.json').read_text())
    def verify(row):
        target=args.dataset/row['video_path']
        if not target.exists():
            if args.source is None and args.source_dataset is None:
                raise FileNotFoundError(target)
            source=((args.source_dataset/row['video_path']) if args.source_dataset else
                    (args.source/row['source_filename'])).resolve(strict=True)
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_suffix('.mp4.tmp')
            try:
                os.link(source,temporary)
            except OSError:
                shutil.copyfile(source,temporary)
            temporary.replace(target)
        if digest(target)!=row['sha256']:
            raise ValueError(f'Video hash mismatch: {target}')
        return dict(episode_id=row['episode_id'],sha256=row['sha256'],bytes=target.stat().st_size)
    with ThreadPoolExecutor(max_workers=8) as pool:
        records=list(pool.map(verify,manifest))
    status=json.loads((args.dataset/'release_status.json').read_text())
    status['video_materialization']='complete'
    status['video_sha256_verification']='passed'
    status['structural_validation']=json.loads((args.dataset/'validation.json').read_text())['structural_validation']
    (args.dataset/'release_status.json').write_text(json.dumps(status,indent=2)+'\n')
    (args.dataset/'video_verification.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(dict(verified_videos=len(records),total_bytes=sum(r['bytes'] for r in records))))


if __name__=='__main__':
    main()
