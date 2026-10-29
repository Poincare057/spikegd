"""Shared destinations and section updates for all QIF outputs."""
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent / 'output'
RESULTS_PATH = OUTPUT_DIR / 'RUN_RESULTS.md'


def write_section(key, title, content, output_path=None):
    """Replace one report section, keeping all other results intact."""
    path = Path(output_path) if output_path is not None else RESULTS_PATH
    start, end = f'<!-- {key}_START -->', f'<!-- {key}_END -->'
    section = f'{start}\n## {title}\n\n{content.rstrip()}\n{end}'
    existing = path.read_text(encoding='utf-8') if path.exists() else '# QIF results\n'
    if start in existing and end in existing:
        before, remainder = existing.split(start, 1)
        _, after = remainder.split(end, 1)
        text = before + section + after
    else:
        text = existing.rstrip() + '\n\n' + section + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


@contextmanager
def capture_results(key, title):
    """Store numerical output in the report instead of separate log files."""
    buffer = StringIO()
    with redirect_stdout(buffer):
        yield
    write_section(key, title, f'```text\n{buffer.getvalue().rstrip()}\n```')
