import matplotlib.figure

import numpy as np
import pytest

from freq import iterative_gls, plot_gls_timeseries
from freq.plot import _instrument_style, annotate, MARKERCYCLE


@pytest.fixture(scope='module')
def multi_inst(synth_rv):
    t, y, yerr, _ = synth_rv
    inst = np.where(t < 100, 'inst_a', 'inst_b')
    res = iterative_gls(t, y, yerr, inst_rv=inst, n=1, plot=True)
    return res


def test_timeseries_from_res_only(multi_inst):
    fig = plot_gls_timeseries(multi_inst)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert fig.axes[-1].get_xlabel() == 'Time [days]'


def test_x_offset_auto_bjd(synth_rv):
    t, y, yerr, _ = synth_rv
    res = iterative_gls(t + 2458800.0, y, yerr, n=1, plot=False)
    fig = plot_gls_timeseries(res)
    assert fig.axes[-1].get_xlabel() == 'BJD $-$ 2457000'


def test_timeseries_yerr_none(synth_rv):
    t, y, _, _ = synth_rv
    res = iterative_gls(t, y, None, n=1, plot=False)
    fig = plot_gls_timeseries(res)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_instrument_style_many_instruments():
    inst = np.array([f'i{k}' for k in range(9)])
    t = np.arange(9.0)
    insts, markers, colors, labels = _instrument_style(inst, t)
    assert len(markers) == 9
    assert all(m in MARKERCYCLE for m in markers)
    assert colors.shape[0] == 9


def test_annotate_bad_loc_raises(multi_inst):
    with pytest.raises(ValueError):
        annotate(multi_inst['fig'].axes[0], 'x', annotate_loc=3)


def test_custom_cmap_respected_single_instrument(synth_rv):
    import matplotlib.pyplot as plt
    t, y, yerr, _ = synth_rv
    res = iterative_gls(t, y, yerr, n=1, plot=False)
    fig = plot_gls_timeseries(res, cmap=plt.cm.viridis)
    sc = [c for c in fig.axes[0].collections if hasattr(c, 'get_cmap')][0]
    assert sc.get_cmap().name == 'viridis'


def test_plot_l1_power_smoke(tmp_path):
    import matplotlib.figure
    from freq.plot import plot_l1_power
    rng = np.random.default_rng(0)
    res = dict(periods=np.geomspace(1, 100, 500),
               power=np.abs(rng.normal(0, 0.1, 500)),
               peak_periods=np.array([5.2, 17.9, 3.3]),
               peak_values=np.array([2.0, 1.0, 0.4]))
    fp = tmp_path / 'l1.png'
    fig = plot_l1_power(res, pmax=50, n_peaks=2, highlight=[5.2], fp=str(fp))
    assert isinstance(fig, matplotlib.figure.Figure)
    assert fp.exists()


def test_plot_l1_power_pmax_excludes_all_raises():
    from freq.plot import plot_l1_power
    res = dict(periods=np.geomspace(10, 100, 50),
               power=np.zeros(50),
               peak_periods=np.array([20.0]),
               peak_values=np.array([1.0]))
    with pytest.raises(ValueError, match='excludes all periods'):
        plot_l1_power(res, pmax=5)
