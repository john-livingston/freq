import os
import subprocess
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, 'data', 'toi1410.txt')
COLS = ['-c', 'btjd', 'rv', 'rv_err', 'inst_name']


def run_cli(args):
    return subprocess.run([sys.executable, '-m', 'freq.cli'] + args,
                          capture_output=True, text=True, cwd=REPO)


def test_gls_run(tmp_path):
    out = str(tmp_path)
    r = run_cli([DATA] + COLS + ['-o', out, '-n', '1', '-mu', '4', '-sm'])
    assert r.returncode == 0, r.stderr
    for f in ('args.txt', 'periodogram.png', 'timeseries.png'):
        assert os.path.exists(os.path.join(out, f)), f


def test_columns_validation(tmp_path):
    r = run_cli([DATA, '-c', 'a', 'b', '-o', str(tmp_path)])
    assert r.returncode != 0
    assert 'exactly 4' in r.stderr


def test_outlier_clip_reported(tmp_path):
    r = run_cli([DATA] + COLS + ['-o', str(tmp_path), '-n', '1', '-oc', '3.5'])
    assert r.returncode == 0, r.stderr
    assert 'outlier clip' in r.stdout


def test_l1_run(tmp_path):
    out = str(tmp_path)
    r = run_cli([DATA] + COLS + ['-o', out, '-n', '1', '-i', 'harpsn', '-sm',
                                 '--l1', '--pmin', '5', '--l1_no_significance'])
    assert r.returncode == 0, r.stderr
    for f in ('l1_periodogram.png', 'l1_peaks.csv', 'l1_periodogram.npz'):
        assert os.path.exists(os.path.join(out, f)), f
    tab = pd.read_csv(os.path.join(out, 'l1_peaks.csv'))
    assert 'period_d' in tab.columns and len(tab) > 0


def test_activity_indicator_missing_column_skipped(tmp_path):
    r = run_cli([DATA] + COLS + ['-o', str(tmp_path), '-n', '1',
                                 '-ai', 'no_such_indicator'])
    assert r.returncode == 0, r.stderr
    assert 'skipping activity indicator' in r.stdout
