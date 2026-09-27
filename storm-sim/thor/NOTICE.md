# Third-party attribution

`vendor/ai2thor/` contains the AI2-THOR Python SDK snapshot used by the source project.
Copyright Allen Institute for Artificial Intelligence and contributors.
Its Apache License 2.0 is retained at `vendor/ai2thor/LICENSE`.
Project: https://github.com/allenai/ai2thor

Local modification: `ai2thor/controller.py` honors the explicit `platform`
argument when `local_executable_path` is supplied, enabling headless
CloudRendering without incorrectly selecting the Linux64/X11 launch environment.

The Unity simulator binaries and scene assets are not bundled. The SDK downloads
them separately, or users can supply an existing executable.

The STORM generation code and new release scripts are separate from this vendored
SDK. No new license for the author's own code is inferred or assigned here.
