"""Save reproducible draft questions alongside completed video captures."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from .natural_qa import build_natural_questions, audit_questions
from .qa import TYPES


def select_draft_questions(pool, event_indices, seed):
    """Choose one question per event, spreading the supported question types."""
    rng = random.Random(seed)
    counts = Counter()
    selected = []
    for index in event_indices:
        candidates = [q for q in pool if q['event_index'] == index]
        if not candidates:
            raise ValueError(f'No evidence-grounded question for event {index}')
        rng.shuffle(candidates)
        question = min(candidates, key=lambda q: counts[q['question_type']])
        selected.append(dict(question, polish_status='pending'))
        counts[question['question_type']] += 1
    return selected


def write_raw_questions(root):
    """Generate annotations from recorded evidence without a model request."""
    root = Path(root)
    events_path = root / 'events.json'
    events = json.loads(events_path.read_text())
    config = json.loads((root / 'config.json').read_text())
    report = json.loads((root / 'observer_validation.json').read_text())
    indices = [event['event_index'] for event in events]
    if len(events) != 10 or len(set(indices)) != 10:
        raise ValueError('Draft QA requires ten distinct completed events')
    episode = config.get('episode_id', root.name)
    pool = build_natural_questions(episode, events, config['seed'])
    questions = select_draft_questions(pool, indices, config['seed'])
    for question in questions:
        if question['query_time'] > report['duration_sec']:
            raise ValueError('Question boundary exceeds the captured video')
    qa = dict(episode_id=episode, video_path='video.mp4',
              duration_sec=report['duration_sec'], sample_fps=report['fps'],
              qa_source='virtualhome_recorded_events', qa_stage='raw',
              questions=questions,
              evaluation_note='Use the video prefix ending at query_time for each question.')
    status = dict(status='raw_ready', polishing='pending', questions=len(questions),
                  candidate_questions=len(pool),
                  events_sha256=hashlib.sha256(events_path.read_bytes()).hexdigest(),
                  question_types=dict(Counter(q['question_type'] for q in questions)),
                  unavailable_candidate_types=sorted(set(TYPES) - {q['question_type'] for q in pool}),
                  selection='One question per event; no model predictions or batch balancing.',
                  audit=audit_questions(questions))
    files = {'qa.json': json.dumps(qa, indent=2),
             'questions.jsonl': ''.join(json.dumps(q) + '\n' for q in questions),
             'qa_candidates.jsonl': ''.join(json.dumps(q) + '\n' for q in pool),
             'qa_status.json': json.dumps(status, indent=2)}
    for name, content in files.items():
        temporary = root / (name + '.tmp')
        temporary.write_text(content)
        temporary.replace(root / name)
    return status
