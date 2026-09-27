"""HTTP client and process lifecycle for the VirtualHome Unity simulator."""
from __future__ import annotations

import base64
import io
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image


class SimulatorError(RuntimeError):
    pass


def xdisplay_numbers(env):
    """Return the bounded X display pool reserved for simulator workers."""
    first = int(env.get('STORM_XDISPLAY_MIN', '190'))
    stop = int(env.get('STORM_XDISPLAY_STOP', '512'))
    if first < 1 or stop <= first or stop > 65536:
        raise ValueError('STORM_XDISPLAY_MIN and STORM_XDISPLAY_STOP define an invalid range')
    return range(first, stop)


class Client:
    def __init__(self, port=8190, timeout=180):
        self.url = f'http://127.0.0.1:{port}'
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers['Connection'] = 'close'

    def command(self, action, *, ints=None, strings=None, timeout=None):
        # The Unity listener needs a frame to reopen after sending a response.
        time.sleep(0.2)
        payload = {'id': str(time.time()), 'action': action}
        if ints is not None:
            payload['intParams'] = ints
        if strings is not None:
            payload['stringParams'] = strings
        # Actions are not retried: a lost response may follow a successful mutation.
        self.session.close()
        response = self.session.post(self.url, json=payload, timeout=timeout or self.timeout)
        response.raise_for_status()
        result = response.json()
        if not result.get('success'):
            raise SimulatorError(f'{action}: {result.get("message", result)}')
        return result

    def reset(self, scene):
        self.command('environment', ints=[scene])

    def graph(self):
        return json.loads(self.command('environment_graph')['message'])

    def character(self, resource='Chars/Male1', room='kitchen', position=None):
        self.command('add_character', strings=[json.dumps({
            'character_resource': resource, 'mode': 'fix_position' if position is not None else 'fix_room',
            'character_position': dict(zip('xyz', position or [0, 0, 0])), 'initial_room': room})])

    def follow_camera(self, position, rotation, fov, name='REAR_OVERHEAD'):
        self.command('add_character_camera', strings=[json.dumps({
            'position': dict(zip('xyz', position)), 'rotation': dict(zip('xyz', rotation)),
            'field_view': fov, 'camera_name': name})])

    def image(self, index, mode='normal', width=640, height=480):
        result = self.command('camera_image', ints=[index], strings=[json.dumps({
            'mode': mode, 'image_width': width, 'image_height': height})])
        return np.array(Image.open(io.BytesIO(base64.b64decode(result['message_list'][0]))).convert('RGB'))

    def colors(self):
        return json.loads(self.command('instance_colors')['message'])

    def render(self, lines, output, prefix, *, camera='REAR_OVERHEAD', fps=20, width=640, height=480,
               record=True, modalities=None, save_pose_data=False, time_scale=1.0, skip_animation=False,
               find_solution=False):
        params = dict(randomize_execution=False, random_seed=0, processing_time_limit=30,
                      skip_execution=False, find_solution=find_solution, output_folder=str(Path(output).resolve()),
                      file_name_prefix=prefix, frame_rate=fps, image_synthesis=modalities or ['normal'],
                      save_pose_data=save_pose_data, save_scene_states=False, camera_mode=[camera], recording=record,
                      image_width=width, image_height=height, time_scale=time_scale,
                      skip_animation=skip_animation)
        return self.command('render_script', strings=[json.dumps(params), *lines])


class UnityProcess:
    def __init__(self, executable, log_dir, *, port=None, gpu_index=0, display=None, xorg_root=None, camera_rules=None):
        self.executable = Path(executable).resolve()
        self.log_dir = Path(log_dir).resolve()
        self.port = port
        self.gpu_index = gpu_index
        self.display = display
        self.xorg_root = Path(xorg_root).resolve() if xorg_root else None
        self.camera_rules = camera_rules
        self.processes = []
        self.handles = []
        self.previous_sigterm = None

    def __enter__(self):
        if not self.executable.is_file():
            raise FileNotFoundError(self.executable)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.previous_sigterm = signal.getsignal(signal.SIGTERM)
        def terminate(signum, frame):
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, terminate)
        if self.port is None:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                self.port = sock.getsockname()[1]
        env = os.environ.copy()
        env['STORM_CAMERA_TRACE'] = str(self.log_dir / 'camera_collision.csv')
        env['STORM_CAMERA_KEEPOUTS'] = str(self.log_dir / 'camera_keepouts.json')
        env['STORM_CAMERA_START'] = str(self.log_dir / 'camera_start')
        env['STORM_CAMERA_VIEW_TARGET'] = str(self.log_dir / 'camera_view_target.json')
        env['STORM_OBSERVER_POSE'] = str(self.log_dir / 'observer_pose.csv')
        if self.camera_rules is not None:
            rules_path = self.log_dir / 'camera_rules.json'
            rules_path.write_text(json.dumps(self.camera_rules))
            env['STORM_CAMERA_RULES'] = str(rules_path)
            env['STORM_OBSERVER_RENDER_ONLY'] = '1'
        try:
            if self.display is None:
                import fcntl
                display_pool=xdisplay_numbers(env)
                for number in display_pool:
                    lease = open(f'/tmp/storm-xdisplay-{number}.lock', 'a')
                    try:
                        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        lease.close()
                        continue
                    if Path(f'/tmp/.X{number}-lock').exists() or Path(f'/tmp/.X11-unix/X{number}').exists():
                        lease.close()
                        continue
                    self.handles.append(lease)
                    self.display = f':{number}'
                    break
                else:
                    raise RuntimeError(
                        f'No unused X display in {display_pool.start}..{display_pool.stop - 1}')
                log = (self.log_dir / 'xvfb.log').open('w')
                self.handles.append(log)
                if self.xorg_root:
                    pci = subprocess.check_output(['nvidia-smi', '-i', str(self.gpu_index),
                                                   '--query-gpu=pci.bus_id', '--format=csv,noheader'], text=True).strip()
                    domain, bus, slot = pci.split(':')
                    device, function = slot.split('.')
                    bus_id = f'PCI:{int(bus, 16)}@{int(domain, 16)}:{int(device, 16)}:{int(function, 16)}'
                    config = self.log_dir / 'xorg.conf'
                    display_mode = env.get('STORM_XORG_DISPLAY_MODE', 'none')
                    if display_mode not in ('none', 'connected'):
                        raise ValueError(f'Unsupported STORM_XORG_DISPLAY_MODE: {display_mode}')
                    display_option = ' Option "UseDisplayDevice" "None"\n' if display_mode == 'none' else ''
                    config.write_text('Section "ServerFlags"\n Option "AutoAddDevices" "False"\n'
                                      ' Option "AutoAddGPU" "False"\nEndSection\n'
                                      'Section "Device"\n Identifier "GPU"\n Driver "nvidia"\n'
                                      f' BusID "{bus_id}"\n Option "AllowEmptyInitialConfiguration" "True"\n'
                                      f'{display_option}EndSection\n'
                                      'Section "Screen"\n Identifier "Screen"\n Device "GPU"\n'
                                      ' DefaultDepth 24\n SubSection "Display"\n Depth 24\n'
                                      ' Virtual 1280 720\n EndSubSection\nEndSection\n')
                    xenv = env.copy()
                    xenv['LD_LIBRARY_PATH'] = str(self.xorg_root / 'usr/lib/x86_64-linux-gnu') + ':' + env.get('LD_LIBRARY_PATH', '')
                    command = [str(self.xorg_root / 'usr/lib/xorg/Xorg'), self.display, '-config', str(config),
                               '-modulepath', str(self.xorg_root / 'usr/lib/xorg/modules') + ',/usr/lib/xorg/modules',
                               '-logfile', str(self.log_dir / 'Xorg.log'), '-noreset', '-nolisten', 'tcp', '-novtswitch']
                else:
                    xenv = env
                    command = ['Xvfb', self.display, '-screen', '0', '1280x720x24', '-nolisten', 'tcp']
                xserver = subprocess.Popen(command, env=xenv, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                self.processes.append(xserver)
                for _ in range(50):
                    if xserver.poll() is not None:
                        raise RuntimeError('X server exited; inspect the display logs')
                    if Path(f'/tmp/.X11-unix/X{self.display[1:]}').exists():
                        break
                    time.sleep(0.1)
                if xserver.poll() is not None:
                    raise RuntimeError('X server exited before Unity startup')
            env['DISPLAY'] = self.display
            log = (self.log_dir / 'unity_stdout.log').open('w')
            self.handles.append(log)
            job_workers = int(env.get('STORM_UNITY_JOB_WORKERS', '8'))
            if not 1 <= job_workers <= 64:
                raise ValueError('STORM_UNITY_JOB_WORKERS must be between 1 and 64')
            args = [str(self.executable), '-screen-fullscreen', '0', '-screen-width', '640', '-screen-height', '480', '-force-glcore', '-job-worker-count', str(job_workers), f'-http-port={self.port}',
                    '-logFile', str(self.log_dir / 'unity.log')]
            proc = subprocess.Popen(args, cwd=self.executable.parent, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            self.processes.append(proc)
            client = Client(self.port)
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f'Unity exited ({proc.returncode}); inspect {self.log_dir}/unity.log')
                try:
                    client.command('idle', timeout=2)
                    self.client = client
                    return client
                except requests.RequestException:
                    time.sleep(1)
            raise TimeoutError('Unity did not start within 120 seconds')
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        if hasattr(self, 'client'):
            self.client.session.close()
        for proc in reversed(self.processes):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
        for handle in self.handles:
            handle.close()
        if self.previous_sigterm is not None:
            signal.signal(signal.SIGTERM, self.previous_sigterm)
