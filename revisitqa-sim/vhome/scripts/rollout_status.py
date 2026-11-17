#!/usr/bin/env python3
"""Report persisted results separately from running or stale worker status."""
import argparse
import json
import os
from pathlib import Path
import time


def summarize(root):
    now=time.time();reports=[]
    for path in sorted(root.glob('*/status.json')):
        data=json.loads(path.read_text())
        if 'rooms' not in data:continue
        accepted=[];inconsistent=[]
        for room in data['rooms']:
            if room.get('status')!='accepted':continue
            directory=Path(room.get('episode',''))
            required=['video.mp4','debug.mp4','qa.json','events.json','event_program.json']
            if all((directory/name).is_file() for name in required):accepted.append(str(directory))
            else:inconsistent.append(str(directory))
        pid=data.get('pid');alive=None
        if pid:
            try:os.kill(pid,0);alive=True
            except ProcessLookupError:alive=False
        reports.append(dict(path=str(path),status=data.get('status'),pid=pid,process_alive=alive,
                            progress_age_seconds=round(now-data.get('progress_at',data.get('updated_at',data.get('started_at',now)))),
                            accepted_artifact_sets=len(accepted),inconsistent_acceptance=inconsistent,
                            rooms=len(data['rooms']),attempts=sum(len(r.get('attempts',[])) for r in data['rooms'])))
    return dict(checked_at=now,reports=reports)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    print(json.dumps(summarize(args.root),indent=2))
