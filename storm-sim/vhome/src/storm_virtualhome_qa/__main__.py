"""Command-line entry points for the standalone QA pipeline."""
import argparse
import sys
from pathlib import Path


def run(argv):
    """Generate raw annotations and apply the established wording revision."""
    from . import generation, revision

    parser = argparse.ArgumentParser(description=run.__doc__)
    for key in ('evidence', 'visibility', 'review', 'reference', 'work', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--quotas', type=Path, default=Path(__file__).with_name('qa290.json'))
    parser.add_argument('--time-limit', type=float, default=90)
    args = parser.parse_args(argv)
    work, output = args.work.resolve(), args.output.resolve()
    if work == output or work in output.parents or output in work.parents:
        parser.error('--work and --output must be separate, non-nested directories')
    for path in (work, output):
        if path.exists():
            raise FileExistsError(path)
    work.mkdir(parents=True)
    generation_args = []
    for key in ('evidence', 'visibility', 'review', 'reference', 'quotas', 'time_limit'):
        generation_args.extend(['--' + key.replace('_', '-'), str(getattr(args, key))])
    generation_args.extend(['--work', str(work / 'selection'), '--output', str(work / 'raw')])
    generation.main(generation_args)
    revision.main(['--source', str(work / 'raw'), '--output', str(output)])


def main(argv=None):
    if not __debug__:
        raise RuntimeError('Run without -O: contract assertions must remain enabled')
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = ('run', 'collect', 'visibility', 'retime', 'generate', 'polish', 'validate')
    parser.add_argument('command', choices=commands)
    if not argv or argv[0] in ('-h', '--help'):
        parser.print_help()
        return
    command = parser.parse_args(argv[:1]).command
    if command == 'run':
        return run(argv[1:])
    from importlib import import_module
    module = {
        'collect': 'collection', 'visibility': 'visibility', 'retime': 'retime',
        'generate': 'generation', 'polish': 'revision', 'validate': 'validation',
    }[command]
    return import_module('.' + module, __package__).main(argv[1:])


if __name__ == '__main__':
    main()
