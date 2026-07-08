import argparse
import os
import sys

import numpy as np
import matplotlib


def build_parser():
    p = argparse.ArgumentParser(
        prog='freq',
        description='freq: frequency analysis of unevenly sampled RV time series')
    g = p.add_argument_group('input')
    g.add_argument('input', help='input file (columns: time rv rv_err instrument)')
    g.add_argument('--sep', default=r'\s+', help='column separator regex')
    g.add_argument('-c', '--columns', nargs='+',
                   default='time mnvel errvel tel'.split(),
                   help='time/rv/err/instrument column names')
    g.add_argument('-o', '--outdir', default='.')
    g = p.add_argument_group('filtering')
    g.add_argument('-mu', '--max_unc', type=float, default=None,
                   help='drop points with rv error above this')
    g.add_argument('-oc', '--outlier_cond', type=float, default=None,
                   help='per-instrument MAD clip factor (before GLS and l1)')
    g.add_argument('-i', '--instruments', nargs='+', default=[],
                   help='only analyze these instruments')
    g = p.add_argument_group('gls')
    g.add_argument('-n', '--n_iter', type=int, default=3)
    g.add_argument('--pmin', type=float, default=1.0)
    g.add_argument('--pmax', type=float, default=100.0)
    g = p.add_argument_group('l1')
    g.add_argument('--l1', action='store_true',
                   help='compute the l1 periodogram instead of the GLS stack')
    g.add_argument('--l1_sigmaW', type=float, default=1.0)
    g.add_argument('--l1_sigmaR', type=float, default=0.0)
    g.add_argument('--l1_tau', type=float, default=0.0)
    g.add_argument('--l1_Prot', type=float, default=-1)
    g.add_argument('--l1_kernel', choices=['gaussian', 'exponential'],
                   default='gaussian',
                   help='red-noise decay kernel (used when --l1_sigmaR > 0)')
    g.add_argument('--l1_qp', choices=['cos', 'ess'], default='cos',
                   help='quasi-periodic factor: cosine bell or exp-sine-squared')
    g.add_argument('--l1_qp_gamma', type=float, default=8.0,
                   help='ess harmonic-content parameter (2/lambda^2)')
    g.add_argument('--l1_trend', action='store_true')
    g.add_argument('--l1_unpenalized', type=float, nargs='+', default=None)
    g.add_argument('--l1_max_tests', type=int, default=12)
    g.add_argument('--l1_no_significance', action='store_true')
    g.add_argument('--l1_n_peaks_plot', type=int, default=4)
    g = p.add_argument_group('l1 cross-validation')
    g.add_argument('--l1_cv', action='store_true',
                   help='rank noise models by CV, then run l1 with the best')
    g.add_argument('--l1_cv_sigmaW', type=float, nargs='+',
                   default=[0.5, 1.0, 2.0])
    g.add_argument('--l1_cv_sigmaR', type=float, nargs='+',
                   default=[0.0, 2.0, 4.0])
    g.add_argument('--l1_cv_tau', type=float, nargs='+', default=[41.0, 82.0])
    g.add_argument('--l1_cv_Prot', type=float, nargs='+', default=[-1.0])
    g.add_argument('--l1_cv_fap_threshold', type=float, default=-0.5)
    g.add_argument('--l1_cv_nsim', type=int, default=400)
    g.add_argument('--l1_cv_training_prop', type=float, default=0.6)
    g.add_argument('--l1_cv_seed', type=int, default=0)
    g.add_argument('--l1_cv_jobs', type=int, default=4)
    g = p.add_argument_group('activity')
    g.add_argument('-ai', '--activity_indicators', nargs='+', default=[])
    g = p.add_argument_group('plotting')
    g.add_argument('-hl', '--highlight', type=float, nargs='+', default=None)
    g.add_argument('--annotate_color', default='k')
    g.add_argument('--x_offset', default='auto',
                   help="'auto' or a number subtracted from plotted times")
    return p


def main(argv=None):
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd

    from .freq import iterative_gls
    from .plot import plot_gls_timeseries, plot_gls_power, FAP_LEVELS
    from .util import ordered_set

    args = build_parser().parse_args(argv)
    if args.l1_cv:
        args.l1 = True
    if len(args.columns) != 4:
        sys.exit(f'-c/--columns needs exactly 4 names '
                 f'(time rv err instrument), got {len(args.columns)}')
    timecol, velcol, errcol, instcol = args.columns

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, 'args.txt'), 'w') as w:
        w.write(' '.join(sys.argv) + '\n')

    df = pd.read_csv(args.input, sep=args.sep, comment='#')

    if args.instruments:
        avail = sorted(df[instcol].unique())
        df = df[df[instcol].isin(args.instruments)]
        if df.empty:
            sys.exit(f'no data for -i {args.instruments}; available: {avail}')

    if args.max_unc is not None:
        idx = df[errcol] > args.max_unc
        print(f'dropping {idx.sum()} points with {errcol} > {args.max_unc}')
        df = df[~idx]

    if args.outlier_cond is not None:
        yv, iv = df[velcol].values, df[instcol].values
        keep = np.ones(len(df), dtype=bool)
        for inst in np.unique(iv):
            ix = iv == inst
            med = np.median(yv[ix])
            mad = np.median(np.abs(yv[ix] - med))
            keep[ix] = np.abs(yv[ix] - med) < args.outlier_cond*mad/0.67
        print(f'outlier clip ({args.outlier_cond} MAD): '
              f'dropping {(~keep).sum()} points')
        df = df[keep]

    x_rv, y_rv, yerr_rv = df[[timecol, velcol, errcol]].values.T
    inst_rv = df[instcol].values

    for inst in ordered_set(inst_rv):
        print(f'{inst}: {(inst_rv == inst).sum()} points')

    if not args.l1:
        x_offset = (args.x_offset if args.x_offset == 'auto'
                    else float(args.x_offset))
        res = iterative_gls(x_rv, y_rv, yerr_rv, inst_rv=inst_rv, n=args.n_iter,
                            pmin=args.pmin, pmax=args.pmax,
                            highlight=args.highlight,
                            annotate_color=args.annotate_color,
                            fp=os.path.join(args.outdir, 'periodogram.png'))
        plot_gls_timeseries(res, x_offset=x_offset,
                            fp=os.path.join(args.outdir, 'timeseries.png'))

    if args.activity_indicators:
        from .gls import Gls
        nrows = len(args.activity_indicators)
        for inst in ordered_set(inst_rv):
            sub = df[df[instcol] == inst]
            fig, axs = plt.subplots(nrows, 1, figsize=(10, 1.5*nrows),
                                    sharex=True, squeeze=False)
            axs = axs.ravel()
            for i, ind in enumerate(args.activity_indicators):
                if ind not in sub.columns:
                    print(f'skipping activity indicator {ind!r}: no such column')
                    axs[i].remove()
                    continue
                errc = f'{ind}_err' if f'{ind}_err' in sub.columns else None
                cols = [timecol, ind] + ([errc] if errc else [])
                vals = sub[cols].dropna().values.T
                if vals.shape[1] == 0 or np.all(vals[1] == 0):
                    axs[i].remove()
                    continue
                if errc and not np.all(vals[2] == 0):
                    data = (vals[0], vals[1], vals[2])
                else:
                    data = (vals[0], vals[1])
                g = Gls(data, Pbeg=args.pmin, Pend=args.pmax)
                plot_gls_power(g, axs[i], fap_levels=FAP_LEVELS,
                               annotate_text=f'{ind}: {g.best["P"] :.1f} days')
                if i < nrows - 1:
                    plt.setp(axs[i], xlabel='', xticklabels='')
            fig.savefig(os.path.join(args.outdir,
                                     f'activity_indicators-{inst}.png'),
                        bbox_inches='tight')

    if args.l1:
        from .l1 import l1_periodogram
        sig = () if args.l1_no_significance else ('fap', 'evidence_laplace')
        if args.l1_cv:
            from .l1cv import l1_crossval
            from .plot import plot_l1_cv_peaks
            cv = l1_crossval(
                x_rv, y_rv, yerr_rv, inst_rv=inst_rv, pmin=args.pmin,
                sigmaW=args.l1_cv_sigmaW, sigmaR=args.l1_cv_sigmaR,
                tau=args.l1_cv_tau, Prot=args.l1_cv_Prot,
                qp=args.l1_qp, qp_gamma=args.l1_qp_gamma,
                kernel=args.l1_kernel,
                fap_threshold=args.l1_cv_fap_threshold,
                n_sim=args.l1_cv_nsim,
                training_prop=args.l1_cv_training_prop,
                seed=args.l1_cv_seed, n_jobs=args.l1_cv_jobs,
                trend=args.l1_trend,
                rerun_kwargs=dict(
                    pmax=args.pmax, significance_methods=sig,
                    max_significance_tests=args.l1_max_tests,
                    n_peaks_plot=args.l1_n_peaks_plot,
                    highlight=args.highlight,
                    annotate_color=args.annotate_color,
                    fp=os.path.join(args.outdir, 'l1_periodogram.png')))
            cvtab = cv['table'].copy()
            for col in ('selected_periods', 'selected_log10faps'):
                cvtab[col] = cvtab[col].map(
                    lambda v: ';'.join(str(x) for x in v))
            cvtab.to_csv(os.path.join(args.outdir, 'l1_cv.csv'), index=False)
            import json
            best = dict(cv['best'], qp=args.l1_qp, kernel=args.l1_kernel,
                        qp_gamma=args.l1_qp_gamma,
                        median_cv=float(cv['table'].iloc[0]['median_cv']))
            with open(os.path.join(args.outdir, 'l1_cv_best.json'), 'w') as w:
                json.dump(best, w, indent=1)
            plot_l1_cv_peaks(cv['table'],
                             fp=os.path.join(args.outdir, 'l1_cv_peaks.png'))
            l1res = cv['l1']
        else:
            l1res = l1_periodogram(
                x_rv, y_rv, yerr_rv, inst_rv=inst_rv,
                pmin=args.pmin, pmax=args.pmax,
                sigmaW=args.l1_sigmaW, sigmaR=args.l1_sigmaR,
                tau=args.l1_tau, Prot=args.l1_Prot, kernel=args.l1_kernel,
                qp=args.l1_qp, qp_gamma=args.l1_qp_gamma,
                trend=args.l1_trend,
                unpenalized_periods=args.l1_unpenalized,
                significance_methods=sig,
                max_significance_tests=args.l1_max_tests,
                n_peaks_plot=args.l1_n_peaks_plot, highlight=args.highlight,
                annotate_color=args.annotate_color,
                fp=os.path.join(args.outdir, 'l1_periodogram.png'))
        l1res['table'].to_csv(os.path.join(args.outdir, 'l1_peaks.csv'),
                              index=False)
        np.savez(os.path.join(args.outdir, 'l1_periodogram.npz'),
                 periods=l1res['periods'], power=l1res['power'],
                 peak_periods=l1res['peak_periods'],
                 peak_values=l1res['peak_values'])

    print('outputs written to', args.outdir)


if __name__ == '__main__':
    main()
