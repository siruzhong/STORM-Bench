#!/usr/bin/env python3
"""Rebuild a captured batch using natural answer domains and metadata-only sampling."""
import argparse
import json
from pathlib import Path
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from storm_virtualhome.natural_qa import build_natural_questions, select_natural_batch, audit_questions


def rebuild(source, output, seed=20260817):
    episodes = sorted((source / 'dataset').glob('episode_*'))
    if not episodes:
        raise ValueError('No captured episodes found')
    pools, paths = [], {}
    for directory in episodes:
        qa = json.loads((directory / 'qa.json').read_text())
        events = json.loads((directory / 'events.json').read_text())
        pool = build_natural_questions(qa['episode_id'], events, seed + len(pools))
        pools.append(pool)
        for q in pool:
            paths[q['id']] = f'dataset/{directory.name}/evaluation/prefixes/{int(q["query_time"] * qa["sample_fps"] + 1e-6):06d}.mp4'
    rows = select_natural_batch(pools, seed)
    for q in rows:
        q['video_path'] = paths[q['id']]
    report = audit_questions(rows)
    report.update(status='development_candidate', selection='Metadata-only; no model outputs used.',
                  video_root=str(source.resolve()),
                  missing_prefixes=sorted({q['video_path'] for q in rows if not (source/q['video_path']).exists()}),
                  release_blockers=['Semantic frame review is pending.',
                                    'Existing event programs still use five paired objects.',
                                    'A held-out text-only baseline and visual controls are pending.'])
    output.mkdir(parents=True, exist_ok=True)
    public = [{k:q[k] for k in ('id','episode_id','question','options','query_time','video_path')} for q in rows]
    labels = [{k:v for k,v in q.items() if k not in ('question','options','video_path')} for q in rows]
    for name, values in [('qa_private.jsonl', rows), ('model_inputs.jsonl', public),
                         ('evaluation_labels.jsonl', labels)]:
        target=output/name
        temp=target.with_suffix('.tmp')
        temp.write_text(''.join(json.dumps(q,ensure_ascii=False)+'\n' for q in values))
        temp.replace(target)
    (output/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    return report


def materialize_prefixes(source, output, workers=4):
    """Create missing prefixes with the original frame rate and exact frame cutoff."""
    import imageio_ffmpeg
    report=json.loads((output/'audit.json').read_text())
    def render(relative):
        destination=source/relative
        video=source/Path(*Path(relative).parts[:2])/'video.mp4'
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_suffix('.tmp.mp4')
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-v','error','-y',
                        '-i',str(video),'-frames:v',str(int(destination.stem)),'-an',
                        '-c:v','libx264','-preset','veryfast','-crf','18','-threads','2',
                        str(temporary)],check=True)
        temporary.replace(destination)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(render,report['missing_prefixes']))
    report['missing_prefixes']=[]
    report['prefixes_materialized']=True
    (output/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20260817)
    parser.add_argument('--render-prefixes',action='store_true')
    args=parser.parse_args()
    report=rebuild(args.input,args.output,args.seed)
    if args.render_prefixes:
        materialize_prefixes(args.input,args.output)
        report=json.loads((args.output/'audit.json').read_text())
    print(json.dumps(report,indent=2,ensure_ascii=False))
