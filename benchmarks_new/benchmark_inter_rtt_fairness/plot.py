#!/usr/bin/env python3
import argparse, os, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
os.environ.setdefault('MPLCONFIGDIR',f'/tmp/matplotlib-{os.getuid()}')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from benchmarks_new.plot_data import aligned_throughput,episode_traces,load_benchmark,metadata_index,number,output_root,return_rows,run_label

def records(root):
    metadata=metadata_index(return_rows(root)); out=[]
    for key,traces in episode_traces(root).items():
        aligned=aligned_throughput(traces)
        if aligned is None: continue
        _,throughput=aligned; active=np.isfinite(throughput); total=np.nansum(throughput,axis=0); squares=np.nansum(throughput**2,axis=0); count=active.sum(axis=0); jain=np.divide(total**2,count*squares,out=np.full_like(total,np.nan),where=(count>1)&(squares>0)); ratios=[]
        for index in range(throughput.shape[1]):
            values=throughput[active[:,index],index]
            if len(values)>1 and np.max(values)>0: ratios.append(np.min(values)/np.max(values))
        meta=metadata.get(key,{}); out.append({'run':key[0],'bw':number(meta,'bw'),'rtt':number(meta,'delay')/2,'jain':np.nanmean(jain),'ratio':np.nanmean(ratios)})
    return out

def heatmap(ax,rows,field,title):
    xs=sorted({r['bw'] for r in rows}); ys=sorted({r['rtt'] for r in rows}); matrix=np.full((len(ys),len(xs)),np.nan)
    for yi,y in enumerate(ys):
        for xi,x in enumerate(xs):
            values=[r[field] for r in rows if r['bw']==x and r['rtt']==y]; matrix[yi,xi]=np.nanmean(values) if values else np.nan
    image=ax.imshow(matrix,origin='lower',aspect='auto',vmin=0,vmax=1,cmap='viridis'); ax.set_xticks(range(len(xs)),[f'{v:g}' for v in xs]); ax.set_yticks(range(len(ys)),[f'{v:g}' for v in ys]); ax.set(xlabel='Bandwidth (Mbps)',ylabel='Flow-2 RTT (ms)',title=title); return image

def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument('--config',default=str(Path(__file__).with_name('config.yaml'))); parser.add_argument('--output'); parser.add_argument('--debug',action='store_true'); args=parser.parse_args(argv); config=Path(args.config).resolve(); manifest,_=load_benchmark(config,debug=args.debug); root=output_root(config,manifest); output=Path(args.output) if args.output else root/'benchmark_summary.pdf'; output.parent.mkdir(parents=True,exist_ok=True); rows=records(root)
    if not rows: print(f'[inter_rtt_plot] no per-flow traces beneath {root}'); return 1
    runs=sorted({r['run'] for r in rows}); fig,axes=plt.subplots(2,len(runs),figsize=(6*len(runs),9),squeeze=False,constrained_layout=True); image=None
    for column,run in enumerate(runs):
        subset=[r for r in rows if r['run']==run]; image=heatmap(axes[0,column],subset,'jain',f'{run_label(run)}\nJain fairness'); heatmap(axes[1,column],subset,'ratio','Minimum / maximum goodput')
    fig.colorbar(image,ax=axes,shrink=.7,label='Fairness ratio'); fig.savefig(output); plt.close(fig); print(f'[inter_rtt_plot] wrote {output}'); return 0
if __name__=='__main__': raise SystemExit(main())
