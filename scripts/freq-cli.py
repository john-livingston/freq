import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("..")
from freq import iterative_gls, plot_gls_timeseries
from freq.util import ordered_set
from freq.plot import plot_gls_power

plt.rcParams['figure.dpi'] = 200

parser = argparse.ArgumentParser(description="freq: Frequency analysis of unevenly sampled time series")

parser.add_argument("input", help="Input file name (with columns: time mnvel errvel tel)")
parser.add_argument("-dw", "--delim_whitespace", help="Delimited by whitespace", action="store_true")
parser.add_argument("-c", "--columns", help="Column names", nargs='+', default="time mnvel errvel tel".split())
parser.add_argument("-o", "--outdir", help="output directory", default=".")
parser.add_argument("-mu", "--max_unc", help="Maximum uncertainty", type=float, default=4)
parser.add_argument("-sm", "--subtract_median", help="Subtract median", action="store_true")
parser.add_argument("-n", "--n_iter", help="Number of iterations", type=int, default=1)
parser.add_argument("--pmin", help="Minimum period", type=float, default=1)
parser.add_argument("--pmax", help="Maximum period", type=float, default=100)
parser.add_argument("-hl", "--highlight", help="Highlight period", type=float, nargs='+', default=[])
parser.add_argument("--annotate_color", help="Annotate color", type=str, default='k')
parser.add_argument("-ai", "--activity_indicators", help="Activity indicators", type=str, nargs='+', default=[])

args = parser.parse_args()
fp = args.input
delim_whitespace = args.delim_whitespace
cols = args.columns
outdir = args.outdir
max_unc = args.max_unc
subtract_median = args.subtract_median
n = args.n_iter
pmin = args.pmin
pmax = args.pmax
highlight = args.highlight
annotate_color = args.annotate_color
activity_indicators = args.activity_indicators

if not os.path.exists(outdir):
    os.mkdir(outdir)

with open(os.path.join(outdir, 'args.txt'), 'w') as w:
    w.write(" ".join(sys.argv)+'\n')

df = pd.read_csv(fp, delim_whitespace=delim_whitespace, comment='#')

timecol = cols[0]
errcol = cols[2]

idx = df[errcol] > max_unc
print(f'dropping {idx.sum()} rv measurements with {errcol} > {max_unc}')
df = df[~idx]

x_rv, y_rv, yerr_rv = df[cols[:3]].values.T
inst_rv = df[cols[3]].values

for i,inst in enumerate(ordered_set(inst_rv)):
    ix = inst_rv == inst
    print(f"{inst} data points: {ix.sum()}")

if subtract_median:
    for i,inst in enumerate(ordered_set(inst_rv)):
        ix = inst_rv == inst
        print(f"{inst} median RV: {np.median(y_rv[ix])}")
        y_rv[ix] -= np.median(y_rv[ix])

res = iterative_gls(
    x_rv, 
    y_rv, 
    yerr_rv, 
    inst_rv=inst_rv, 
    n=n, 
    pmin=pmin, 
    pmax=pmax, 
    highlight=highlight, 
    annotate_color=annotate_color,
    fp=os.path.join(outdir, 'periodogram.png')
)

plot_gls_timeseries(
    res, 
    x_rv, 
    y_rv, 
    yerr_rv, 
    inst_rv=inst_rv,
    fp=os.path.join(outdir, 'timeseries.png')
)

if len(activity_indicators) > 0:
    
    from gls import Gls

    for inst in ordered_set(inst_rv):
        ix = inst_rv == inst
        
        nrows = len(activity_indicators)
        fig, axs = plt.subplots(nrows, 1, figsize=(10,1.5*nrows), sharex=True)
        if nrows == 1: axs = [axs]
        gls = []
        for i,(ind,ind_name) in enumerate(zip(activity_indicators, activity_indicators)):
            ax = axs[i]
            if f'{ind}_err' in df.columns:
                cols = f'{timecol} {ind} {ind}_err'.split()
                x, y, yerr = df.loc[ix,cols].dropna().values.T
                data = (x,y,yerr)
                if not np.isfinite(yerr).any() or np.all(yerr == 0):
                    data = (x,y)
            else:
                cols = f'{timecol} {ind}'.split()
                x, y = df.loc[ix,cols].values.T
                data = (x,y)
            if not np.isfinite(y).any() or np.all(y == 0):
                axs[i].remove()
                gls.append(None)
                continue
            gls.append(Gls(data, Pbeg=pmin, Pend=pmax))
            
            plot_gls_power(gls[i], ax, fap_levels=[1e-1,1e-2,1e-3],
                        annotate_text=f'{ind_name}: {gls[i].best["P"] :.1f} days')
            if i < nrows-1: plt.setp(ax, xlabel='', xticklabels='')
        plt.savefig(os.path.join(outdir, f'activity_indicators-{inst}.png'), bbox_inches='tight')
