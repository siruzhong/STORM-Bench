#!/usr/bin/env python3
"""Check local dependencies; optionally launch one scene and save an RGB frame."""
import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render', action='store_true', help='Actually start AI2-THOR; may download Unity on first run')
    parser.add_argument('--output', type=Path, default=Path('outputs/environment_smoke.png'))
    args = parser.parse_args()
    packages = {}
    for name in ('numpy', 'PyYAML', 'Pillow', 'imageio', 'imageio-ffmpeg', 'ai2thor'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = 'not installed'
    import imageio_ffmpeg
    print(json.dumps({'python': sys.version.split()[0], 'platform': platform.platform(),
                      'packages': packages, 'ffmpeg': imageio_ffmpeg.get_ffmpeg_exe()}, indent=2))
    if args.render:
        from PIL import Image
        from tools.storm.benchgen.adapters.ai2thor import Ai2ThorSession
        from tools.storm.benchgen.domain.config import load_config
        session = Ai2ThorSession()
        try:
            event = session.reset('FloorPlan1', load_config(ROOT / 'configs/benchgen_qa_4scene_v1.yaml'))
            if not event.metadata.get('lastActionSuccess') or event.frame.shape != (480, 640, 3):
                raise RuntimeError('simulator smoke render failed')
            args.output.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(event.frame).save(args.output)
            print(json.dumps({'render': 'passed', 'rgb_shape': list(event.frame.shape),
                              'objects': len(event.metadata['objects']), 'image': str(args.output.resolve())}))
        finally:
            session.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
