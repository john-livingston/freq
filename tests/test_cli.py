import io
import json
import os
import re
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import pandas as pd

from freq.cli import main as cli_main

COLS = ['-c', 'btjd', 'rv', 'rv_err', 'inst_name']


class _CliResult:
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def run_cli(args):
    """Call freq.cli.main in-process so coverage sees the CLI branches."""
    import matplotlib.pyplot as plt
    out, err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            cli_main(args)
    except SystemExit as e:
        code = e.code
        extra = ''
        if code is None:
            rc = 0
        elif isinstance(code, int):
            rc = code
        else:
            rc = 1
            extra = str(code) + '\n'
        return _CliResult(rc, out.getvalue(), err.getvalue() + extra)
    finally:
        plt.close('all')
    return _CliResult(0, out.getvalue(), err.getvalue())


def test_module_entrypoint(tmp_path, rv_file):
    """python -m freq.cli is the installed console path.

    Catches: dropping `if __name__ == '__main__'` so the module imports but
    does not run.
    """
    r = subprocess.run(
        [sys.executable, '-m', 'freq.cli', rv_file, *COLS,
         '-o', str(tmp_path), '-n', '1'],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert os.path.exists(os.path.join(str(tmp_path), 'periodogram.png'))


def test_gls_run(tmp_path, rv_file):
    out = str(tmp_path)
    r = run_cli([rv_file] + COLS + ['-o', out, '-n', '1', '-mu', '4'])
    assert r.returncode == 0, r.stderr
    for f in ('args.txt', 'periodogram.png', 'timeseries.png'):
        assert os.path.exists(os.path.join(out, f)), f
    for f in ('l1_periodogram.png', 'l1_peaks.csv'):
        assert not os.path.exists(os.path.join(out, f)), f


def test_sm_flag_removed(tmp_path, rv_file):
    r = run_cli([rv_file] + COLS + ['-o', str(tmp_path), '-n', '1', '-sm'])
    assert r.returncode != 0


def test_columns_validation(tmp_path, rv_file):
    r = run_cli([rv_file, '-c', 'a', 'b', '-o', str(tmp_path)])
    assert r.returncode != 0
    assert 'exactly 4' in r.stderr


def test_outlier_clip_drops_exactly_the_outliers(tmp_path, rv_file_messy):
    """The MAD clip drops the injected outliers and nothing else.

    Catches: a wrong threshold (e.g. mad*0.67 instead of mad/0.67), which a
    smoke test asserting only that the run printed something cannot see.
    """
    r = run_cli([rv_file_messy] + COLS + ['-o', str(tmp_path), '-n', '1',
                                          '-oc', '3.5'])
    assert r.returncode == 0, r.stderr
    m = re.search(r'outlier clip \(3\.5 MAD\): dropping (\d+) points', r.stdout)
    assert m, r.stdout
    assert int(m.group(1)) == 2          # the two injected +/-500 m/s points


def test_l1_run(tmp_path, rv_file):
    out = str(tmp_path)
    r = run_cli([rv_file] + COLS + ['-o', out, '-i', 'harpsn',
                                    '--l1', '--pmin', '5', '--l1_no_significance'])
    assert r.returncode == 0, r.stderr
    for f in ('l1_periodogram.png', 'l1_peaks.csv', 'l1_periodogram.npz'):
        assert os.path.exists(os.path.join(out, f)), f
    for f in ('periodogram.png', 'timeseries.png'):
        assert not os.path.exists(os.path.join(out, f)), f
    tab = pd.read_csv(os.path.join(out, 'l1_peaks.csv'))
    assert 'period_d' in tab.columns and len(tab) > 0


def test_activity_indicator_missing_column_skipped(tmp_path, rv_file):
    r = run_cli([rv_file] + COLS + ['-o', str(tmp_path), '-n', '1',
                                    '-ai', 'no_such_indicator'])
    assert r.returncode == 0, r.stderr
    assert 'skipping activity indicator' in r.stdout


def test_l1_cv_run(tmp_path, rv_file):
    out = str(tmp_path)
    r = run_cli([rv_file] + COLS + ['-o', out, '-i', 'harpsn', '--l1_cv',
                                    '--pmin', '5', '--l1_no_significance',
                                    '--l1_cv_sigmaW', '1',
                                    '--l1_cv_sigmaR', '0', '2',
                                    '--l1_cv_tau', '10', '--l1_cv_Prot', '-1',
                                    '--l1_cv_nsim', '30', '--l1_cv_jobs', '1'])
    assert r.returncode == 0, r.stderr
    for f in ('l1_cv.csv', 'l1_cv_peaks.png', 'l1_periodogram.png',
              'l1_peaks.csv', 'l1_cv_best.json'):
        assert os.path.exists(os.path.join(out, f)), f
    best = json.load(open(os.path.join(out, 'l1_cv_best.json')))
    assert set(best) >= {'sigmaW', 'sigmaR', 'tau', 'Prot'}
    assert not os.path.exists(os.path.join(out, 'periodogram.png'))


def test_activity_indicators_report_periods(tmp_path, rv_file_activity):
    """Each usable indicator's best period is reported; all-zero ones skipped.

    Catches: dropna() removal (NaNs -> nan period), wrong error-column
    selection, and removal of the all-zero guard.
    """
    out = str(tmp_path)
    r = run_cli([rv_file_activity, '-c', 'btjd', 'rv', 'rv_err', 'inst_name',
                 '-o', out, '-n', '1', '--pmin', '2', '--pmax', '40',
                 '-ai', 'shk', 'halpha', 'zeroerr', 'flat'])
    assert r.returncode == 0, r.stderr
    # periods recovered from the indicator time series themselves
    assert re.search(r'shk:\s*11\.[45]', r.stdout), r.stdout
    assert re.search(r'halpha:\s*6\.[34]', r.stdout), r.stdout
    assert re.search(r'zeroerr:\s*8\.[12]', r.stdout), r.stdout
    assert 'flat:' not in r.stdout            # identically zero -> skipped
    for inst in ('carmenes', 'harpsn'):
        assert os.path.exists(
            os.path.join(out, f'activity_indicators-{inst}.png')), inst


def test_zero_mad_instrument_survives_outlier_clip(tmp_path, rv_file_messy):
    """A constant-velocity instrument must not be wiped out by the MAD clip.

    Catches: threshold cond*mad/0.67 collapsing to 0 when mad == 0, which
    discards every point of that instrument.
    """
    out = str(tmp_path)
    r = run_cli([rv_file_messy] + COLS + ['-o', out, '-n', '1', '-oc', '3.5'])
    assert r.returncode == 0, r.stderr
    assert re.search(r'inst_c: \d+ points', r.stdout), r.stdout
    n = int(re.search(r'inst_c: (\d+) points', r.stdout).group(1))
    assert n > 0


def test_non_finite_rows_dropped_and_reported(tmp_path, rv_file_messy):
    """NaN rv/rv_err rows are dropped before fitting, with a count.

    Catches: NaNs reaching Gls, which makes the whole power array NaN and
    yields a meaningless best period with FAP nan and exit code 0.
    """
    out = str(tmp_path)
    r = run_cli([rv_file_messy] + COLS + ['-o', out, '-n', '1'])
    assert r.returncode == 0, r.stderr
    assert re.search(r'dropping 2 non-finite', r.stdout), r.stdout
    assert 'nan' not in r.stdout.lower()


def _stub_l1_result():
    tab = pd.DataFrame({'period_d': [5.2], 'amplitude': [1.0]})
    return dict(table=tab, periods=np.array([5.2]), power=np.array([1.0]),
                peak_periods=np.array([5.2]), peak_values=np.array([1.0]))


def _stub_cv_result():
    tab = pd.DataFrame([{
        'sigmaW': 1.0, 'sigmaR': 0.0, 'tau': 10.0, 'Prot': -1.0,
        'median_cv': -100.0,
        'selected_periods': [], 'selected_log10faps': [],
    }])
    return dict(table=tab,
                best={'sigmaW': 1.0, 'sigmaR': 0.0, 'tau': 10.0, 'Prot': -1.0},
                l1=_stub_l1_result())


def test_sep_reads_comma_separated_file(tmp_path, synth_rv, monkeypatch):
    """--sep ',' is passed to read_csv, so a real CSV is readable.

    Catches: --sep being ignored, so the default whitespace regex treats each
    comma-separated line as a single column and KeyError-exits on -c names.
    """
    t, y, yerr, _ = synth_rv
    p = tmp_path / 'rv.csv'
    with open(p, 'w') as w:
        w.write('btjd,rv,rv_err,inst_name\n')
        for ti, yi, ei in zip(t, y, yerr):
            inst = 'a' if ti < 100 else 'b'
            w.write(f'{ti:.6f},{yi:.6f},{ei:.6f},{inst}\n')
    monkeypatch.setattr('freq.core.iterative_gls', lambda *a, **k: {})
    monkeypatch.setattr('freq.plot.plot_gls_timeseries', lambda *a, **k: None)
    from freq.cli import main
    main([str(p), '--sep', ',', *COLS, '-o', str(tmp_path / 'out'), '-n', '1'])


def test_cli_forwards_gls_highlight_and_x_offset(tmp_path, rv_file, monkeypatch):
    """-hl, --annotate_color, and --x_offset reach iterative_gls / the timeseries plot.

    Catches: the GLS call dropping highlight, or x_offset staying the string
    'auto' / never being float()ed when a number is given.
    """
    captured = {}
    monkeypatch.setattr(
        'freq.core.iterative_gls',
        lambda *a, **k: captured.update(gls=k) or {})
    monkeypatch.setattr(
        'freq.plot.plot_gls_timeseries',
        lambda *a, **k: captured.update(ts=k))
    from freq.cli import main
    main([rv_file, *COLS, '-o', str(tmp_path), '-n', '1',
          '-hl', '5.2', '11.0', '--x_offset', '2457000',
          '--annotate_color', 'r'])
    assert captured['gls']['highlight'] == [5.2, 11.0]
    assert captured['gls']['annotate_color'] == 'r'
    assert captured['ts']['x_offset'] == 2457000.0


def test_cli_forwards_l1_noise_and_null_model(tmp_path, rv_file, monkeypatch):
    """--l1 kernel/qp/trend/unpenalized/highlight reach l1_periodogram.

    Catches: argparse accepting the flags but main() never passing them, the
    same class of miss as unpenalized_periods used to have under --l1_cv.
    """
    captured = {}

    def fake(*a, **k):
        captured.update(k)
        return _stub_l1_result()

    monkeypatch.setattr('freq.l1.l1_periodogram', fake)
    from freq.cli import main
    main([rv_file, *COLS, '-o', str(tmp_path), '--l1', '--l1_no_significance',
          '--l1_kernel', 'exponential', '--l1_qp', 'ess', '--l1_qp_gamma', '4',
          '--l1_sigmaR', '2', '--l1_tau', '10', '--l1_Prot', '30',
          '--l1_trend', '--l1_unpenalized', '5.234', '-hl', '7.5'])
    assert captured['kernel'] == 'exponential'
    assert captured['qp'] == 'ess'
    assert captured['qp_gamma'] == 4.0
    assert captured['sigmaR'] == 2.0
    assert captured['tau'] == 10.0
    assert captured['Prot'] == 30.0
    assert captured['trend'] is True
    assert captured['unpenalized_periods'] == [5.234]
    assert captured['highlight'] == [7.5]
    assert captured['significance_methods'] == ()


def test_cli_cv_forwards_unpenalized_trend_kernel(tmp_path, rv_file, monkeypatch):
    """--l1_cv forwards --l1_unpenalized, --l1_trend, kernel/qp, and --l1_no_significance.

    Catches: the flags being wired only on the non-CV l1 branch, including
    --l1_no_significance landing only in rerun_kwargs['significance_methods'].
    """
    captured = {}

    def fake(*a, **k):
        captured.update(k)
        return _stub_cv_result()

    monkeypatch.setattr('freq.l1cv.l1_crossval', fake)
    from freq.cli import main
    main([rv_file, *COLS, '-o', str(tmp_path), '--l1_cv', '--l1_no_significance',
          '--l1_trend', '--l1_unpenalized', '5.234',
          '--l1_kernel', 'exponential', '--l1_qp', 'ess',
          '--l1_cv_sigmaW', '1', '--l1_cv_sigmaR', '0',
          '--l1_cv_tau', '10', '--l1_cv_Prot', '-1',
          '--l1_cv_nsim', '10', '--l1_cv_jobs', '1'])
    assert captured['unpenalized_periods'] == [5.234]
    assert captured['trend'] is True
    assert captured['kernel'] == 'exponential'
    assert captured['qp'] == 'ess'
    assert captured['rerun_kwargs']['significance_methods'] == ()


def test_unknown_instrument_exits(tmp_path, rv_file):
    """-i with no matching instrument exits before fitting.

    Catches: an empty slice proceeding into GLS and producing a cryptic
    numpy error instead of the available-instrument message.
    """
    r = run_cli([rv_file, *COLS, '-o', str(tmp_path), '-i', 'not_an_inst',
                 '-n', '1'])
    assert r.returncode != 0
    assert 'no data for -i' in r.stderr
    assert 'carmenes' in r.stderr


def test_all_non_finite_rows_exit(tmp_path):
    """A file whose every RV/error is NaN exits after the drop.

    Catches: the empty-after-dropna path falling through to Gls, which
    then yields an all-NaN periodogram with exit 0.
    """
    p = tmp_path / 'nan.txt'
    p.write_text('btjd rv rv_err inst_name\n1 nan 1 a\n2 1 nan a\n')
    r = run_cli([str(p), *COLS, '-o', str(tmp_path / 'out'), '-n', '1'])
    assert r.returncode != 0
    assert 'no usable rows' in r.stderr
