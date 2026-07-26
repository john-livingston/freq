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


def test_plot_l1_power_annotates_top_peaks():
    from freq.plot import plot_l1_power
    res = dict(periods=np.geomspace(1, 100, 500),
               power=np.zeros(500),
               peak_periods=np.array([5.2, 17.9, 3.3]),
               peak_values=np.array([2.0, 1.0, 0.4]))
    fig = plot_l1_power(res, n_peaks=2)
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert any('5.20' in s for s in texts)
    assert any('17.90' in s for s in texts)
    assert not any('3.30' in s for s in texts)


def test_plot_l1_cv_peaks_smoke(tmp_path):
    import pandas as pd
    from freq.plot import plot_l1_cv_peaks
    tab = pd.DataFrame([
        dict(median_cv=-100.0, selected_periods=[5.2, 17.9],
             selected_log10faps=[-3.0, -1.0]),
        dict(median_cv=-110.0, selected_periods=[],
             selected_log10faps=[]),
    ])
    fp = tmp_path / 'cv.png'
    fig = plot_l1_cv_peaks(tab, perc=50, fp=str(fp))
    assert fig is not None and fp.exists()


def test_log_period_axis_plain_number_labels():
    from freq.plot import _log_period_axis
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([2, 5, 20], [1, 2, 3])
    _log_period_axis(ax)
    ax.set_xlim(4, 22)
    fmt = ax.xaxis.get_major_formatter()
    assert fmt(10.0) == '10'      # not '$10^1$' or '10^{1}'
    assert fmt(20.0) == '20'
    assert fmt(2.0) == '2'


def _vline_positions(ax):
    """x positions of vertical lines (axvline gives xdata == (x, x))."""
    out = []
    for ln in ax.lines:
        xd = ln.get_xdata()
        if len(xd) == 2 and xd[0] == xd[1]:
            out.append(float(xd[0]))
    return out


def test_highlight_draws_vlines_on_gls_power(synth_rv):
    """-hl/highlight periods are marked on the GLS periodogram.

    Catches: the highlight loop being dropped -> user's marked periods
    never appear on the plot.
    """
    import matplotlib.pyplot as plt
    from freq.plot import plot_gls_power
    t, y, yerr, _ = synth_rv
    res = iterative_gls(t, y, yerr, n=1, plot=False)
    fig, ax = plt.subplots()
    plot_gls_power(res['gls'][0], ax, highlight=[3.5, 12.25])
    pos = _vline_positions(ax)
    assert 3.5 in pos and 12.25 in pos


def test_highlight_draws_vlines_on_l1_power():
    """highlight periods are marked on the l1 periodogram too."""
    from freq.plot import plot_l1_power
    res = dict(periods=np.geomspace(1, 100, 300), power=np.zeros(300),
               peak_periods=np.array([5.0]), peak_values=np.array([1.0]))
    fig = plot_l1_power(res, highlight=[7.5])
    assert 7.5 in _vline_positions(fig.axes[0])


def test_folded_plot_without_instruments(synth_rv):
    """The single-instrument folded panel plots points and error bars.

    Catches: the inst_rv=None branch losing its scatter or errorbar call.
    """
    import matplotlib.pyplot as plt
    from freq.plot import plot_gls_folded
    t, y, yerr, _ = synth_rv
    res = iterative_gls(t, y, yerr, n=1, plot=False)
    from matplotlib.collections import LineCollection, PathCollection
    fig, ax = plt.subplots()
    plot_gls_folded(res['gls'][0], ax, yerr=True, inst_rv=None)
    scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
    bars = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert len(scatters) == 1                          # the folded data points
    assert len(scatters[0].get_offsets()) == len(t)    # every point plotted
    assert len(bars) == 1                              # error bars drawn
    assert len(ax.lines) >= 1                          # sinusoid model curve
