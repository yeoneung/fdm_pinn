"""Render numerical tables and figures from computed summaries."""
from pathlib import Path
import shutil
from . import common, periodic, budget, policy, slices, air_and_perturbations, finite_controls


def main(source, output):
    output.mkdir(parents=True, exist_ok=False)
    for name in ['data', 'tables', 'figures']:
        (output/name).mkdir()
    for path in Path(source).glob('*.json'):
        shutil.copy2(path, output/'data'/path.name)
    common.HERE = output
    for module in [periodic, budget, policy, slices, air_and_perturbations, finite_controls]:
        module.HERE = output
        module.main()
    print(f"Generated {len(list((output/'tables').glob('*.tex')))} tables and "
          f"{len(list((output/'figures').glob('*.pdf')))} figures in {output}")
