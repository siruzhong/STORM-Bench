from __future__ import annotations

import copy
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

import imageio.v2 as imageio
import numpy as np

from scripts.dataset_io import load_documents, read_json, validate_bundle, write_json, write_jsonl
from scripts.polish_qa import apply_response, extract_frames, main, make_messages, sample_times, validate_response


def question():
    return dict(id='episode_q1', episode_id='episode', query_time=0.9, question_type='state_change',
                question_subtype='fixture', question='Which object disappears?',
                options=['mug', 'book', 'plate', 'bowl'], answer_index=0,
                video_evidence='The mug is visible before the change and absent afterward.',
                evidence_spans=[[0, 0.9]], diagnostics={'epistemic_status': 'known', 'uncertainty_sources': []},
                diagnostic_rationale={'volatility': 'fixture', 'uncertainty': 'fixture'}, change_intensity=1)


def response(q, **overrides):
    return dict(id=q['id'], decision='rewrite', question='Which listed object disappears from view?',
                video_evidence=q['video_evidence'], reason='Language edit only.', **overrides)


def fixture(root):
    source = root / 'source'
    video = source / 'meta_data/gene_videos/FloorPlan1/episode.mp4'
    video.parent.mkdir(parents=True)
    with imageio.get_writer(str(video), fps=10, codec='libx264', macro_block_size=16) as writer:
        for i in range(20):
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            frame[:, :, 0 if i < 10 else 1] = 220
            writer.append_data(frame)
    document = dict(episode_id='episode', video_path='/old/computer/episode.mp4', duration_sec=2,
                    sample_fps=2, sample_timestamps=[0, 0.5, 0.9, 1, 1.5], qa_source='test', questions=[question()])
    write_json(source / 'meta_data/qa_results/FloorPlan1/episode.json', document)
    write_jsonl(source / 'questions.jsonl', document['questions'])
    return source, document, video


class PolishTests(unittest.TestCase):
    def test_request_uses_packaged_wording_prompt(self):
        expected = files('tools.storm').joinpath('prompts/qa_polish.txt').read_text(encoding='utf-8').strip()
        messages = make_messages(question(), [])
        self.assertEqual(messages[0], {'role': 'system', 'content': expected})
        self.assertTrue(expected)

    def test_only_two_text_fields_can_change(self):
        q = question()
        result = validate_response(q, json.dumps(response(q)))
        updated = apply_response(q, result)
        self.assertEqual({k: v for k, v in updated.items() if k not in ('question', 'video_evidence')},
                         {k: v for k, v in q.items() if k not in ('question', 'video_evidence')})
        for forbidden in ('answer_index', 'options', 'query_time', 'diagnostics'):
            bad = dict(result, **{forbidden: 'modified'})
            with self.assertRaises(ValueError):
                validate_response(q, json.dumps(bad))

    def test_flag_keeps_original_and_numeric_cues_rejected(self):
        q = question()
        result = response(q)
        result['decision'] = 'flag'
        self.assertEqual(apply_response(q, result), q)
        result['decision'] = 'rewrite'
        result['question'] = 'What happens at 42 seconds?'
        with self.assertRaises(ValueError):
            validate_response(q, json.dumps(result))

    def test_actual_decoded_frames_never_show_future(self):
        from PIL import Image
        import io
        with tempfile.TemporaryDirectory() as tmp:
            _, document, video = fixture(Path(tmp))
            times = sample_times(document, question(), 12)
            self.assertEqual(times, [0, 0.5, 0.9])
            frames = extract_frames(video, times, 0.9)
            for frame in frames:
                pixels = np.array(Image.open(io.BytesIO(frame['jpeg'])))
                self.assertGreater(pixels[:, :, 0].mean(), pixels[:, :, 1].mean() + 100)
                self.assertLessEqual(frame['timestamp'], 0.9)
            with self.assertRaises(ValueError):
                extract_frames(video, [1.5], 0.9)

    def test_http_retry_resume_and_portable_bundle(self):
        calls = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                calls.append(payload)
                if len(calls) == 1:
                    self.send_response(503)
                    self.end_headers()
                    return
                text = json.dumps(response(question()))
                body = json.dumps({'choices': [{'message': {'content': text}}]}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {'VLM_API_KEY': 'test-only-key'}):
                source, _, _ = fixture(Path(tmp))
                output = Path(tmp) / 'polished'
                argv = ['--input', str(source), '--output', str(output), '--model', 'mock-vision',
                        '--base-url', f'http://127.0.0.1:{server.server_port}/v1', '--retries', '1']
                original = (source / 'questions.jsonl').read_bytes()
                self.assertEqual(main(argv), 0)
                self.assertEqual(len(calls), 2)
                self.assertTrue(any(c['type'] == 'image_url' for c in calls[-1]['messages'][1]['content']))
                self.assertEqual(main(argv + ['--resume']), 0)
                self.assertEqual(len(calls), 2)
                self.assertEqual((source / 'questions.jsonl').read_bytes(), original)
                exported = load_documents(output)[0][1]['questions'][0]
                self.assertEqual(exported['question'], response(question())['question'])
                self.assertEqual(exported['answer_index'], 0)
                record = json.loads((output / 'model_inputs.jsonl').read_text())
                self.assertNotIn('video_evidence', json.dumps(record))
                self.assertNotIn('test-only-key', (output / 'polish_run.json').read_text())
                moved = Path(tmp) / 'moved'
                output.rename(moved)
                self.assertEqual(validate_bundle(moved)['question_count'], 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_invalid_api_output_falls_back_and_resume_can_retry(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {'VLM_API_KEY': 'test-only-key'}):
            source, _, _ = fixture(Path(tmp))
            output = Path(tmp) / 'polished'
            argv = ['--input', str(source), '--output', str(output), '--model', 'mock', '--base-url', 'http://127.0.0.1/v1']
            with patch('scripts.polish_qa.call_vlm', return_value=json.dumps({'answer_index': 3})):
                self.assertEqual(main(argv), 2)
            self.assertEqual(load_documents(output)[0][1]['questions'][0], question())
            with patch('scripts.polish_qa.call_vlm', return_value=json.dumps(response(question()))) as call:
                self.assertEqual(main(argv + ['--resume', '--retry-failed']), 0)
                self.assertEqual(call.call_count, 1)
            with self.assertRaises(SystemExit):
                main(argv + ['--resume', '--max-frames', '4'])

    def test_dry_run_never_calls_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, _, _ = fixture(Path(tmp))
            output = Path(tmp) / 'dry'
            with patch('scripts.polish_qa.call_vlm', side_effect=AssertionError('must not call API')):
                self.assertEqual(main(['--input', str(source), '--output', str(output), '--dry-run']), 0)
            self.assertFalse((output / 'questions.jsonl').exists())
            self.assertEqual(len(list((output / 'requests').glob('*.json'))), 1)

    def test_future_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, document, _ = fixture(Path(tmp))
            bad = copy.deepcopy(document)
            bad['questions'][0]['evidence_spans'] = [[0, 1.5]]
            write_json(source / 'meta_data/qa_results/FloorPlan1/episode.json', bad)
            with self.assertRaises(ValueError):
                load_documents(source)


if __name__ == '__main__':
    unittest.main()
