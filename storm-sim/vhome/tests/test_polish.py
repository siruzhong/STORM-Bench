"""Exercise the VLM adapter against a local endpoint and a two-second video."""
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.resources import files
import io
import json
from pathlib import Path
import subprocess
import sys
import threading

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def test_endpoint_receives_only_prefix_and_preserves_labels(tmp_path, monkeypatch):
    video = tmp_path / 'video.mp4'
    with imageio.get_writer(str(video), fps=2, codec='libx264', macro_block_size=1) as writer:
        for color in ([255, 0, 0], [255, 0, 0], [0, 0, 255], [0, 0, 255]):
            writer.append_data(np.full((32, 32, 3), color, dtype=np.uint8))
    question = {'id': 'q1', 'query_time': 1.0, 'question': 'What is visible?',
                'options': ['red', 'blue', 'green', 'yellow'], 'answer_index': 0,
                'video_evidence': 'PRIVATE', 'evidence_spans': [[0, 1]]}
    (tmp_path / 'qa.json').write_text(json.dumps({'video_path': 'video.mp4', 'questions': [question]}))
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            body = json.dumps({'choices': [{'message': {'content': json.dumps({'question': 'Which color is visible?'})}}]}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv('TEST_VLM_KEY', 'local-test')
    monkeypatch.setenv('NO_PROXY', '127.0.0.1')
    try:
        script = Path(__file__).resolve().parents[1] / 'scripts/polish_qa.py'
        subprocess.run([sys.executable, str(script), '--input', str(tmp_path / 'qa.json'),
                        '--output', str(tmp_path / 'polished'), '--endpoint', f'http://127.0.0.1:{server.server_port}',
                        '--model', 'test', '--api-key-env', 'TEST_VLM_KEY', '--frames', '2'], check=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert len(received) == 1
    expected_prompt = files('storm_virtualhome').joinpath('prompts/qa_polish.txt').read_text(encoding='utf-8').strip()
    assert received[0]['messages'][0] == {'role': 'system', 'content': expected_prompt}
    assert 'PRIVATE' not in json.dumps(received)
    text = json.loads(received[0]['messages'][1]['content'][0]['text'])
    assert set(text) == {'question', 'options', 'query_time'}
    images = [x for x in received[0]['messages'][1]['content'] if x['type'] == 'image_url']
    assert len(images) == 2
    for item in images:
        image = np.array(Image.open(io.BytesIO(base64.b64decode(item['image_url']['url'].split(',')[1]))))
        assert image[:, :, 0].mean() > 200
        assert image[:, :, 2].mean() < 30
    output = json.loads((tmp_path / 'polished/qa.json').read_text())['questions'][0]
    assert output['question'] == 'Which color is visible?'
    for key in ('options', 'answer_index', 'video_evidence', 'evidence_spans', 'query_time'):
        assert output[key] == question[key]
