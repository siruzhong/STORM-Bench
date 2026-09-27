"""Check review isolation and API failures without a network service."""
import copy
import math
import pytest

from storm_virtualhome.judgment import (
    failure_request, plan_request, qa_request, review, validate_response, rewrite_supported,
)


def question():
    return dict(id='q1', question='What is the current state of the lamp?',
                original_question='At the end of the clip, is the lamp on or off?',
                options=['on', 'off'], answer_index=1, query_time=12,
                diagnostics={'epistemic_status': 'known'},
                hidden_event={'verb': 'SwitchOff'}, video_evidence='secret label')


def response_for(request):
    answers = {}
    for key, item in request['questions'].items():
        if item['type'] == 'noul':
            answers[key] = dict(type='noul', noul=.7)
        else:
            options = list(item['criteria'])
            answers[key] = dict(type='choice', choice=options[0], confidence=1,
                                probabilities={x: float(x == options[0]) for x in options})
    return dict(model='jev-test', answers=answers, usage={})


def test_qa_request_cannot_read_labels_or_hidden_state():
    row = question()
    before = copy.deepcopy(row)
    request = qa_request(row)
    assert set(request['state']) == {'question', 'options', 'original_question'}
    assert row == before
    assert 'preserves_knowledge' in request['questions']


def test_no_original_means_no_equivalence_claim():
    row = question()
    del row['original_question']
    assert not any(x.startswith('preserves_') for x in qa_request(row)['questions'])


@pytest.mark.parametrize('mutation', ['missing', 'extra_option', 'nan', 'wrong_choice', 'bad_sum'])
def test_malformed_response_is_unavailable(mutation):
    request = qa_request(question())
    response = response_for(request)
    answer = response['answers']['option_overlap']
    if mutation == 'missing':
        del response['answers']['reference_clarity']
    elif mutation == 'extra_option':
        answer['probabilities']['invented'] = 0
    elif mutation == 'nan':
        answer['confidence'] = math.nan
    elif mutation == 'wrong_choice':
        answer['choice'] = 'overlap'
    else:
        answer['probabilities']['distinct'] = .7
    result = review(request, transport=lambda _: response)
    assert result['status'] == 'unavailable'
    assert result['automatic_action'] == 'none'
    assert 'response' in result


def test_timeout_does_not_leak_error_message_or_change_policy():
    def timeout(_):
        raise TimeoutError('secret token must not be logged')
    result = review(qa_request(question()), transport=timeout)
    assert result['status'] == 'unavailable'
    assert result['error_type'] == 'TimeoutError'
    assert 'secret token' not in str(result)
    assert result['automatic_action'] == 'none'


def test_dry_run_never_calls_transport():
    def unexpected(_):
        raise AssertionError('Unexpected network call')
    assert review(qa_request(question()), dry_run=True, transport=unexpected)['status'] == 'prepared'


def test_valid_response_retains_raw_probabilities():
    request = failure_request({'failure': 'Patrol exceeded the frozen window for event 0'})
    response = response_for(request)
    validate_response(request, response)
    assert review(request, transport=lambda _: response)['response'] == response


def test_missing_failure_and_empty_plan_rejected():
    with pytest.raises(ValueError):
        failure_request({'failure': ''})
    with pytest.raises(ValueError):
        plan_request({'events': []})


def test_plan_does_not_request_physics_acceptance():
    request = plan_request(dict(room='kitchen', events=[dict(verb='Grab', object='mug')]))
    assert set(request['questions']) == {'room_fit', 'routine_coherence'}


def test_rewrite_requires_every_check_and_sufficient_confidence():
    request = qa_request(question())
    result = review(request, transport=response_for)
    assert rewrite_supported(result)
    result['response']['answers']['preserves_time']['confidence'] = .5
    assert not rewrite_supported(result)
    result['status'] = 'unavailable'
    assert not rewrite_supported(result)


def test_plan_keeps_actor_but_ignores_unused_surface():
    request = plan_request(dict(room='bathroom', events=[
        dict(verb='SwitchOn', object='lightswitch', injector_character=2, surface='rug')]))
    event = request['state']['events'][0]
    assert event['injector_character'] == 2
    assert 'surface' not in event
