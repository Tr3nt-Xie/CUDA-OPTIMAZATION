#!/usr/bin/env python3
"""Presentation figures from the verified C run; does not run benchmarks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id', default='C-20261013-full-02')
    args = p.parse_args()
    raw = ROOT / 'results/raw' / args.run_id
    summary = ROOT / 'results/summary' / args.run_id
    out = ROOT / 'results/figures' / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    coverage = json.loads((summary / 'coverage.json').read_text())
    assert coverage['status'] == 'complete'
    rows = list(csv.DictReader((summary / 'summary.csv').open(newline='', encoding='utf-8')))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'axes.labelcolor': '#26364b',
                         'text.color': '#1c2e45', 'axes.edgecolor': '#b9c4d3',
                         'figure.facecolor': '#f8fafc', 'axes.facecolor': 'white',
                         'savefig.facecolor': '#f8fafc', 'svg.fonttype': 'none'})
    names = {'cpu': 'CPU (single thread)', 'naive': 'Naive CUDA', 'tiled': 'Tiled CUDA',
             'optimized': 'Optimized CUDA', 'cublas': 'cuBLAS FP32',
             'python_tiled': 'Python + tiled', 'cuda': 'CUDA', 'python_cuda': 'Python + CUDA'}
    colors = {'cpu': '#64748b', 'naive': '#e99b26', 'tiled': '#306ee8', 'optimized': '#0e9384',
              'cublas': '#b43c87', 'python_tiled': '#8563c7', 'cuda': '#306ee8', 'python_cuda': '#0e9384'}
    generated = []

    def save(fig, name):
        for ext in ('png', 'svg'):
            fig.savefig(out / f'{name}.{ext}', dpi=170)
        generated.append(name)
        plt.close(fig)

    def format_x(ax, xs):
        ax.set_xscale('log', base=2)
        ax.set_xticks(xs, labels=[str(x) for x in xs])
        ax.minorticks_off()
        ax.grid(which='major', alpha=.18)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.4))
    fig.subplots_adjust(top=.76, bottom=.25, left=.075, right=.98, wspace=.22)
    fig.text(.06, .945, 'LAB6  /  MATRIX MULTIPLICATION', fontsize=21, weight='bold')
    fig.text(.06, .885, 'AWS NVIDIA T4  |  CUDA 12.8  |  FP32  |  1 warm-up + 3 measured runs', fontsize=11, color='#536880')
    fig.text(.06, .825, '24 matrix configurations verified  /  all recorded results passed tolerance', fontsize=11, color='#0e766b')
    for ax, scope, title in zip(axes, ('compute','end_to_end'), ('Compute time', 'Complete call time')):
        for impl in ('cpu','naive','tiled','optimized','cublas','python_tiled'):
            selected = sorted((r for r in rows if r['operation']=='matrix' and r['implementation']==impl and r['timing_scope']==scope), key=lambda r:int(r['N']))
            ax.errorbar([int(r['N']) for r in selected], [float(r['mean_ms']) for r in selected],
                        yerr=[float(r['std_ms']) for r in selected], label=names[impl],
                        color=colors[impl], marker='s' if impl=='python_tiled' else 'o',
                        linestyle='--' if impl=='python_tiled' else '-', linewidth=1.9, markersize=5, capsize=3)
        ax.set(title=title, ylabel='Mean runtime (ms, log scale)', xlabel='Matrix side length N', yscale='log')
        format_x(ax, [256,512,1024,2048])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.52,.06), ncol=3, frameon=False)
    fig.text(.06,.027, 'Bars: sample standard deviation. Complete call includes GPU allocation, transfers and cleanup; cuBLAS also includes handle lifecycle.', fontsize=8.7, color='#536880')
    save(fig, 'matrix-runtime')

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6))
    fig.subplots_adjust(top=.80, bottom=.22, left=.08, right=.98, wspace=.22)
    fig.suptitle('LAB6  /  MATRIX SPEEDUP OVER SINGLE-THREAD CPU', x=.06, ha='left', fontsize=19, weight='bold')
    for ax, scope in zip(axes, ('compute','end_to_end')):
        for impl in ('naive','tiled','optimized','cublas','python_tiled'):
            selected = sorted((r for r in rows if r['operation']=='matrix' and r['implementation']==impl and r['timing_scope']==scope), key=lambda r:int(r['N']))
            ax.plot([int(r['N']) for r in selected], [float(r['cpu_speedup']) for r in selected],
                    color=colors[impl], label=names[impl], marker='o', linestyle='--' if impl=='python_tiled' else '-')
        ax.set(title=scope, ylabel='CPU mean / implementation mean (log scale)', xlabel='Matrix side length N', yscale='log')
        format_x(ax,[256,512,1024,2048])
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.53,.05), ncol=3, frameon=False)
    fig.text(.06,.018, 'Ratios use identical input, host and timing scope. The CPU baseline is the lab triple-loop implementation, not an optimized BLAS library.', fontsize=9, color='#536880')
    save(fig,'matrix-speedup')

    for scope in ('compute','end_to_end'):
        fig, axes = plt.subplots(3,3,figsize=(13.6,10.8),sharex=True)
        fig.subplots_adjust(top=.86,bottom=.12,left=.08,right=.98,hspace=.46,wspace=.30)
        fig.text(.06,.955,'LAB6  /  CONVOLUTION SCALING',fontsize=21,weight='bold')
        fig.text(.06,.918,f'{scope}  |  3 images x 3 filter sizes x 3 image sizes  |  mean filter, zero padding, true convolution',fontsize=11,color='#536880')
        for rowidx,img in enumerate(('geometry','frequency','soft_scene')):
            for col,k in enumerate((3,5,7)):
                ax=axes[rowidx,col]
                for impl in ('cpu','cuda','python_cuda'):
                    selected=sorted((r for r in rows if r['operation']=='convolution' and r['input_id'].startswith(img+':') and r['K']==str(k) and r['implementation']==impl and r['timing_scope']==scope),key=lambda r:int(r['M']))
                    ax.errorbar([int(r['M']) for r in selected],[float(r['mean_ms']) for r in selected],
                                yerr=[float(r['std_ms']) for r in selected],color=colors[impl],label=names[impl],
                                marker='s' if impl=='python_cuda' else 'o',linestyle='--' if impl=='python_cuda' else '-',capsize=3,linewidth=1.7)
                ax.set(title=f'{img}  /  K={k}',yscale='log')
                format_x(ax,[512,1024,2048])
                if col==0: ax.set_ylabel('Mean runtime (ms, log)')
                if rowidx==2: ax.set_xlabel('Image side length M')
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.52,.045),ncol=3,frameon=False)
        fig.text(.06,.018,'Mean of all 3 measured calls; bars show sample SD. Native and Python measurements were collected sequentially.',fontsize=9,color='#536880')
        save(fig,f'convolution-{scope}')

    fig,axes=plt.subplots(1,4,figsize=(14,4.6))
    fig.subplots_adjust(top=.80,bottom=.19,left=.025,right=.975,wspace=.045)
    fig.text(.035,.925,'LAB6  /  REAL PYTHON + CUDA FILTER OUTPUTS',fontsize=20,weight='bold')
    files=[ROOT/'data/images/geometry.png', *[raw/'python-filter-demo'/f'{name}.png' for name in ('mean','edge','sharpen')]]
    for ax,path,label in zip(axes,files,('Original','Mean (3 x 3)','Edge (absolute display)','Sharpen')):
        with Image.open(path) as im: ax.imshow(im.copy(),cmap='gray',vmin=0,vmax=255)
        ax.set_title(label,fontsize=12,pad=10)
        ax.axis('off')
    fig.text(.035,.10,'All three outputs passed independent reference checks. Signed float32 arrays are saved before display conversion.',fontsize=10,color='#536880')
    fig.text(.035,.055,'Edge PNG uses absolute magnitude; display values are rounded and clipped to [0, 255]. Numerical validation uses the original signed output.',fontsize=9,color='#536880')
    save(fig,'python-filter-demo')
    provenance={'run_id':args.run_id,'summary_sha256':hashlib.sha256((summary/'summary.csv').read_bytes()).hexdigest(),
                'plot_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'figures':generated,
                'note':'Figures added after benchmark completion; benchmark source snapshot remains unchanged.'}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(out)


if __name__=='__main__':
    main()
