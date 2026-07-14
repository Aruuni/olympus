#!/usr/bin/env python3
import argparse, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault('MPLCONFIGDIR', f'/tmp/matplotlib-{os.getuid()}')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from benchmarks_new.plot_data import aligned_throughput, episode_traces, load_benchmark, output_root, run_label, state_rows

def ecdf(ax, values, label):
    values=np.sort(np.asarray(values,float)[np.isfinite(values)])
    if values.size: ax.plot(values,np.arange(1,values.size+1)/values.size*100,label=label,linewidth=1.5)

def fairness(root):
    rows=[]
    for (run, episode), traces in episode_traces(root).items():
        aligned=aligned_throughput(traces)
        if aligned is None: continue
        _, throughput=aligned; active=np.isfinite(throughput); total=np.nansum(throughput,axis=0); squares=np.nansum(throughput**2,axis=0); count=active.sum(axis=0)
        jain=np.divide(total**2,count*squares,out=np.full_like(total,np.nan),where=(count>1)&(squares>0))
        if np.isfinite(jain).any(): rows.append((run,float(np.nanmean(jain))))
    return rows

def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument('--config',default=str(Path(__file__).with_name('config.yaml'))); parser.add_argument('--output'); parser.add_argument('--debug',action='store_true'); args=parser.parse_args(argv); config=Path(args.config).resolve(); manifest,_=load_benchmark(config,debug=args.debug); root=output_root(config,manifest); output=Path(args.output) if args.output else root/'benchmark_summary.pdf'; output.parent.mkdir(parents=True,exist_ok=True); states=state_rows(root); fair=fairness(root)
    if not states: print(f'[responsive_fairness_plot] no traces beneath {root}'); return 1
    fig,axes=plt.subplots(1,3,figsize=(16,5.5),constrained_layout=True)
    for run in sorted({r['run'] for r in states}):
        subset=[r for r in states if r['run']==run]; label=run_label(run); ecdf(axes[0],[r['throughput'] for r in subset],label); ecdf(axes[1],[r['srtt'] for r in subset],label); ecdf(axes[2],[value for name,value in fair if name==run],label)
    for ax,title,label in zip(axes,('Goodput CDF','SRTT CDF','Jain Fairness CDF'),('Average per-flow goodput (Mbps)','Average SRTT (ms)','Jain index')):
        ax.set(title=title,xlabel=label,ylabel='CDF (%)',ylim=(0,100)); ax.grid(alpha=.25); ax.legend(frameon=False)
    fig.savefig(output); plt.close(fig); print(f'[responsive_fairness_plot] wrote {output}'); return 0
if __name__=='__main__': raise SystemExit(main())
