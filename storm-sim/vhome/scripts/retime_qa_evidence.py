"""Correct observation timestamps using stable-state visibility in recorded frames."""
import copy
import json
import math
from pathlib import Path
import sys

root=Path(sys.argv[1])
bundle=json.loads((root/'evidence_bundle.json').read_text())
visibility={x['episode_id']:x for x in json.loads((root/'visibility_audit.json').read_text())['episodes']}
report=[]
for ep in bundle['episodes']:
    v=visibility[ep['metadata']['episode_id']]
    fps=v['fps']
    source_events=copy.deepcopy(ep['events'])
    records={(x['event_index'],x['side']):x for x in ep['observations']}
    for e in ep['events']:
        col=v['objects'].index(e['object_id'])
        threshold=12 if e['object']=='lightswitch' else 40
        previous=max([x['end'] for x in source_events if x['object_id']==e['object_id'] and x['end']<e['start']]+[0])
        following=min([x['start'] for x in source_events if x['object_id']==e['object_id'] and x['start']>e['end']]+[v['frames']/fps])
        ranges={'before':(math.ceil(previous*fps-1e-6),math.ceil(e['start']*fps-1e-6)),
                'after':(math.ceil(e['end']*fps-1e-6),min(math.ceil(following*fps-1e-6),v['frames']))}
        for side,(lo,hi) in ranges.items():
            o=records[e['event_index'],side]
            frames=[i for i in range(lo,hi) if v['counts'][i][col]>=threshold]
            if not frames:
                o['state']=None
                continue
            frame=frames[-1] if side=='before' else frames[0]
            o['original_stage_time']=o['time']
            o['original_stage_frame']=o['frame']
            o.update(frame=frame,time=(frame+1)/fps,pixels=v['counts'][frame][col],
                     visibility_threshold=threshold,
                     state_interval=[lo/fps,hi/fps],
                     visibility_source='visibility_audit.json',
                     frame_manifest_sha256=v['manifest_sha256'])
            e['before_time' if side=='before' else 'discovery_time']=o['time']
        report.append(dict(episode=ep['metadata']['episode_id'],event=e['event_index'],
                           old_discovery=source_events[e['event_index']]['discovery_time'],
                           corrected_discovery=e['discovery_time']))
    ep['original_events']=source_events
    ep['timeline_source']='Graph-verified stable states and full-frame instance-mask visibility.'
(root/'retimed_evidence_bundle.json').write_text(json.dumps(bundle))
summary=dict(events=len(report),changed=sum(abs(x['old_discovery']-x['corrected_discovery'])>.051 for x in report),
             more_than_one_second=sum(abs(x['old_discovery']-x['corrected_discovery'])>1 for x in report),
             max_abs_seconds=max(abs(x['old_discovery']-x['corrected_discovery']) for x in report),rows=report,
             qualification='Instance masks establish object visibility, not perceptual readability of a state.')
(root/'timeline_corrections.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='rows'}))
