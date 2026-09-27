"""Reject missing or modified code when an exact-source replay is requested."""
import hashlib
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from replay_native_captures import frozen_source


def test_exact_source_is_verified_against_attempt_and_file_hashes(tmp_path):
    hashes={'src/example.py':hashlib.sha256(b'original').hexdigest()}
    revision=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:12]
    source=tmp_path/'sources'/revision
    (source/'src').mkdir(parents=True);(source/'scripts').mkdir()
    file=source/'src/example.py';file.write_bytes(b'original')
    (source/'source_hashes.json').write_text(json.dumps(hashes))
    attempt=tmp_path/'attempts'/'01';capture=attempt/'capture';capture.mkdir(parents=True)
    (tmp_path/'status.json').write_text(json.dumps(dict(rooms=[dict(attempts=[
        dict(path=str(attempt),source_revision=revision)])])))
    assert frozen_source(capture)==(source,revision)
    file.write_bytes(b'modified')
    with pytest.raises(ValueError,match='missing or changed'):
        frozen_source(capture)


def test_missing_attempt_provenance_is_not_replaced_with_current_code(tmp_path):
    with pytest.raises(ValueError,match='No attempt provenance'):
        frozen_source(tmp_path/'capture')
