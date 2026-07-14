#!/usr/bin/env python3
import argparse, os, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
os.environ.setdefault('MPLCONFIGDIR',f'/tmp/matplotlib-{os.getuid()}')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from benchmarks_new.common import load_scenario
from benchmarks_new.plot_data import aligned_throughput, episode_traces, load_benchmark, metadata_index, number, output_root, return_rows, run_label

def join_times(config, manifest, overlay):
    scenario_value=(manifest.get('matrix') or {}).get('scenarios',[None])[0]
    if not scenario_value: return []
    path=Path(scenario_value); path=path if path.is_absolute() else config.parent/path
    resolved=load_scenario(path,overlay); sweep=resolved.get('sweep') or {}; flow_schedule=sweep.get('flow_schedule') or {}; arrival=flow_schedule.get('arrival') or {}
    if arrival.get('start_delays') is not None: return [float(value) for value in arrival['start_delays']]
    if arrival.get('evenly_spaced_over_s') is not None:
        flows=(sweep.get('flows') or [1])[0]; window=float(arrival['evenly_spaced_over_s']); return [window*i/max(int(flows)-1,1) for i in range(int(flows))]
    return []

def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument('--config',default=str(Path(__file__).with_name('config.yaml'))); parser.add_argument('--output'); parser.add_argument('--debug',action='store_true'); args=parser.parse_args(argv); config=Path(args.config).resolve(); manifest,overlay=load_benchmark(config,debug=args.debug); joins=join_times(config,manifest,overlay); root=output_root(config,manifest); output=Path(args.output) if args.output else root/'benchmark_summary.pdf'; output.parent.mkdir(parents=True,exist_ok=True); metadata=metadata_index(return_rows(root)); records=[]
    for key,traces in episode_traces(root).items():
        aligned=aligned_throughput(traces); meta=metadata.get(key,{})
        if aligned is None or not np.isfinite(number(meta,'bw')) or number(meta,'bw')<=0: continue
        t,throughput=aligned; active=np.isfinite(throughput); total=np.nansum(throughput,axis=0); squares=np.nansum(throughput**2,axis=0); count=active.sum(axis=0); jain=np.ones_like(total); np.divide(total**2,count*squares,out=jain,where=(count>1)&(squares>0)); records.append((key[0],t,total/number(meta,'bw'),jain))
    if not records: print(f'[convergence_plot] no per-flow traces beneath {root}'); return 1
    fig,axes=plt.subplots(2,1,figsize=(12,9),sharex=True,constrained_layout=True)
    for run in sorted({r[0] for r in records}):
        subset=[r for r in records if r[0]==run]; end=min(r[1][-1] for r in subset); grid=np.arange(0,end+1e-9); util=np.vstack([np.interp(grid,r[1],r[2]) for r in subset]); fair=np.vstack([np.interp(grid,r[1],r[3]) for r in subset]); label=run_label(run); axes[0].plot(grid,np.nanmean(util,axis=0),label=label); axes[1].plot(grid,np.nanmean(fair,axis=0),label=label)
    axes[0].set(ylabel='Total goodput / capacity',title='Capacity utilization'); axes[1].set(xlabel='Episode time (s)',ylabel='Jain index',title='Flow convergence fairness',ylim=(0,1.05))
    for ax in axes:
        for join in joins[1:]: ax.axvline(join,color='black',linestyle=':',linewidth=.8,alpha=.5)
        ax.grid(alpha=.25); ax.legend(frameon=False)
    fig.savefig(output); plt.close(fig); print(f'[convergence_plot] wrote {output}'); return 0
if __name__=='__main__': raise SystemExit(main())
