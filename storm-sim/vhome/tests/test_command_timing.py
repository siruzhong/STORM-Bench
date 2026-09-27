"""Keep native preparation separate from the actual event start window."""
import pytest
from storm_virtualhome.planning import operation_command_time, check_injection_start


def test_open_close_lead_uses_their_longer_native_manipulation():
    event=dict(event_index=1,verb='Close',planned_start_seconds=12.65,
               start_tolerance_seconds=2.5,planned_gap_seconds=7.03,
               measured_operation_seconds=4.95)
    config=dict(event_interval_seconds=6,event_interval_tolerance_seconds=2.5)
    command,lead=operation_command_time(event,12.65,5.1,config)
    assert lead==pytest.approx(1.7)
    assert command==pytest.approx(10.4)
    check_injection_start(event,command+lead,5.1,config)


def test_variable_preparation_delay_cannot_cause_an_early_event():
    event=dict(event_index=1,verb='SwitchOn',planned_start_seconds=12,
               start_tolerance_seconds=2.5,planned_gap_seconds=6,
               measured_operation_seconds=3.3)
    config=dict(event_interval_seconds=6,event_interval_tolerance_seconds=2.5)
    command,lead=operation_command_time(event,10,7,config)
    assert command==pytest.approx(10.5)
    check_injection_start(event,command,7,config)
    check_injection_start(event,command+lead,7,config)


def test_first_event_leaves_room_for_a_slower_second_action():
    event=dict(event_index=0,verb='Open',planned_start_seconds=5.65,
               start_tolerance_seconds=2.5,measured_operation_seconds=3.3)
    config=dict(event_interval_seconds=6,event_interval_tolerance_seconds=2.5)
    command,_=operation_command_time(event,5.65,None,config)
    assert command==pytest.approx(4.4)
    check_injection_start(event,command,None,config)
    check_injection_start(event,command+2,None,config)
