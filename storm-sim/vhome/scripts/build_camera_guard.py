#!/usr/bin/env python3
"""Build a separate VirtualHome runtime with a collision-aware follow camera."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('.runtime/simulator'))
    parser.add_argument('--output', type=Path, default=Path('.runtime/simulator-debug'))
    parser.add_argument('--mono-root', type=Path, default=Path('.runtime/mono'))
    parser.add_argument('--observer-walk-pacing', action='store_true',
                        help='Enable experimental observer animation pacing for joint actions')
    args = parser.parse_args()
    source, output, mono_root = args.source.resolve(), args.output.resolve(), args.mono_root.resolve()
    if output.exists():
        parser.error('Output exists; use a new directory')
    managed_relative = Path('linux_exec.v2.3.0_Data/Managed')
    original = source / managed_relative / 'Assembly-CSharp.dll'
    if not original.is_file():
        parser.error('Expected the VirtualHome 2.3.0 Linux runtime')
    cecil = next((mono_root / 'usr/lib/mono/gac/Mono.Cecil').glob('0.11*/Mono.Cecil.dll'))
    mono = mono_root / 'usr/bin/mono-sgen'
    compiler = mono_root / 'usr/lib/mono/4.5/mcs.exe'
    env = os.environ.copy()
    env.update(MONO_CFG_DIR=str(mono_root / 'etc'), LD_LIBRARY_PATH=str(mono_root / 'usr/lib'),
               MONO_PATH=str(mono_root / 'usr/lib/mono/4.5') + ':' + str(cecil.parent))
    shutil.copytree(source, output)
    managed = output / managed_relative
    sources = Path(__file__).resolve().parents[1] / 'unity_debug'
    addon = managed / 'StormCameraGuard.dll'
    build = output / '.build'
    build.mkdir()
    patcher = build / 'PatchCamera.exe'
    subprocess.run([str(mono), str(compiler), '-target:library', f'-out:{addon}',
                    f'-r:{managed / "UnityEngine.CoreModule.dll"}',
                    f'-r:{managed / "UnityEngine.PhysicsModule.dll"}',
                    f'-r:{managed / "UnityEngine.AIModule.dll"}',
                    f'-r:{managed / "UnityEngine.AnimationModule.dll"}',
                    f'-r:{managed / "Newtonsoft.Json.dll"}',
                    f'-r:{managed / "netstandard.dll"}',
                    str(sources / 'SafeFollowCamera.cs'),
                    str(sources / 'NativeNavigation.cs')], env=env, check=True)
    subprocess.run([str(mono), str(compiler), f'-out:{patcher}', f'-r:{cecil}',
                    str(sources / 'PatchCamera.cs')], env=env, check=True)
    patched = managed / 'Assembly-CSharp.patched.dll'
    patch_command = [str(mono), str(patcher), str(original), str(addon), str(patched)]
    if args.observer_walk_pacing:
        patch_command.append('--observer-walk-pacing')
    subprocess.run(patch_command, env=env, check=True)
    patched.replace(managed / 'Assembly-CSharp.dll')
    manifest = {'source_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
                'patched_sha256': hashlib.sha256((managed / 'Assembly-CSharp.dll').read_bytes()).hexdigest(),
                'guard_source_sha256': hashlib.sha256((sources / 'SafeFollowCamera.cs').read_bytes()).hexdigest(),
                'navigation_source_sha256': hashlib.sha256((sources / 'NativeNavigation.cs').read_bytes()).hexdigest(),
                'native_navigation': True,
                'supported_viewpoints': ['first_person','third_person'],
                'first_person_attachment_evidence': True,
                'per_frame_actor_positions': True,
                'observer_joint_walk_pacing': args.observer_walk_pacing,
                'native_action_clock': True,
                'native_sight_probe': True,
                'collision_radius_m': 0.18, 'clearance_m': 0.04, 'near_clip_m': 0.06,
                'pivot_height_m': 1.9, 'focus_height_m': 1.5, 'preferred_minimum_arm_m': 1.1,
                'controller': 'continuous', 'smoothing': 'damped_position_and_angles', 'fixed_view_direction': True, 'clock': 'recorded_observer_frames',
                'character_line_of_sight_obstacles': False, 'check_observer_overlap': True,
                'scene_resource_fallback': 'existing_scene_prefab_by_exact_name'}
    (output / 'camera_guard.json').write_text(json.dumps(manifest, indent=2))
    print(output / 'linux_exec.v2.3.0.x86_64')


if __name__ == '__main__':
    main()
