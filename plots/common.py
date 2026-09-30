"""Formatting and summary-statistics checks for numerical outputs."""
from pathlib import Path


import json


import numpy as np


import matplotlib


matplotlib.use('Agg')


import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent


METHODS = ['AD_direct', 'FD_direct', 'AD_PI', 'FD_PI']


NAMES = ['Direct PINN', 'FD-direct', 'AD-PINN-PI', 'FD-PINN-PI']


COLORS = ['#3567A8', '#D87926', '#487E63', '#8B5AA8']


MARKERS = ['o', 's', '^', 'D']


def read(name):
    return json.loads((HERE / 'data' / (name + '.json')).read_text(encoding='utf-8'))


def pm(d, factor=1, digits=3):
    return f"${factor*d['mean']:.{digits}f} \\pm {factor*d['std']:.{digits}f}$"


def table(name, spec, header, rows):
    out = '\\begin{tabular}{' + spec + '}\n\\toprule\n' + header + ' \\\\\n\\midrule\n'
    out += '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n'
    (HERE / 'tables' / (name + '.tex')).write_text(out, encoding='utf-8')


def check_summary(d):
    """Recompute stored means/SDs from the individual values actually plotted."""
    count = 0
    if isinstance(d, dict):
        if {'mean', 'std', 'values'} <= d.keys():
            values = np.asarray(d['values'], dtype=float)
            assert np.isclose(values.mean(), d['mean'], rtol=1e-10, atol=1e-14)
            assert np.isclose(values.std(ddof=1), d['std'], rtol=1e-10, atol=1e-14)
            count += 1
        for val in d.values():
            count += check_summary(val)
    elif isinstance(d, list):
        count += sum(check_summary(val) for val in d)
    return count
