"""Keep native probes balanced across interaction families."""
from storm_virtualhome.room_preflight import probe_candidates, probe_result_ready, switch_animation_ready


def test_probe_candidates_reserve_slots_for_each_action_family():
    candidates=[]
    for key in range(8):
        candidates.append(dict(id=key,name='mug',operations=['Grab'],unique_mask=True,bounds={'size':[.1,.1,.1]}))
    candidates.extend([
        dict(id=20,name='faucet',operations=['SwitchOn'],unique_mask=True,bounds={'size':[.1,.1,.1]}),
        dict(id=21,name='cabinet',operations=['Open'],unique_mask=True,bounds={'size':[1,1,1]}),
    ])
    selected=probe_candidates({'candidates':candidates},limit=6)
    assert selected[0]['id']==21
    assert [candidate['probe_family'] for candidate in selected[:3]]==['Open','SwitchOn','Grab']
    assert {operation for candidate in selected for operation in candidate['operations']}=={'Grab','SwitchOn','Open'}


def test_probe_success_requires_target_and_verb_diversity():
    def target(key, name, first, second, accepted=True):
        return dict(id=key,name=name,operations=[dict(first=first,second=second,accepted=accepted)])
    targets=[target(1,'cabinet','Open','Close'),target(2,'drawer','Open','Close'),
             target(3,'computer','SwitchOn','SwitchOff')]
    assert probe_result_ready(targets)
    assert not probe_result_ready(targets[:2])
    assert not probe_result_ready([*targets[:2],target(4,'closet','Open','Close')])
    assert not probe_result_ready([*targets[:2],target(5,'lamp','SwitchOn','SwitchOff',False)])
    assert probe_result_ready([targets[0],target(6,'cabinet','Open','Close'),targets[2]])


def test_switch_animation_requires_visible_target_change():
    timing=dict(recorded_frames=12,target_shared_pixels=80,target_mean_frame_delta=3.5)
    assert switch_animation_ready(timing)
    assert not switch_animation_ready(dict(timing,target_shared_pixels=0,mean_frame_delta=20))
    assert not switch_animation_ready(dict(timing,target_mean_frame_delta=.1))
    assert not switch_animation_ready(dict(timing,recorded_frames=1))
