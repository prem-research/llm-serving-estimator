import numpy as np, pandas as pd
from servest import calib, perf
P=perf.load_params()
F,Lb,M=calib.load('primary'); pr=calib.predict(F,P)
out=perf.solve(F,P)
for t in calib.TARGETS:
    m=Lb[t].notna()&(Lb[t]>0)&np.isfinite(pr[t])&(pr[t]>0)
    d=M[m].copy(); d['e']=np.log(pr[t][m]/Lb[t][m]).values; d['rho']=out['prefill_util'][m.values]
    tot=d.e.var()
    print(f'\n{t}: total var {tot:.3f} (sd {np.sqrt(tot):.2f})')
    for k in ['family','gpu','engine','scheme','group']:
        within=d.groupby(k)['e'].transform(lambda x:x-x.mean()).var()
        print(f'  explained by {k:7s}: {1-within/tot:5.1%}  -> residual sd {np.sqrt(within):.2f}')
    d['fg']=d.family+'|'+d.gpu
    within=d.groupby('fg')['e'].transform(lambda x:x-x.mean()).var(); print(f'  explained by family x gpu: {1-within/tot:5.1%} -> sd {np.sqrt(within):.2f}')
    if t=='ttft_ms':
        d['rb']=pd.cut(d.rho,[-.01,.1,.3,.6,.9,1]); print('  by prefill util', d.groupby('rb',observed=True)['e'].agg(['median','count']).round(2).to_dict('index'))
    g=d.groupby('family')['e'].agg(['median','count']).sort_values('median')
    print('  worst families:', g[g['count']>30].iloc[[0,1,2,-3,-2,-1]].round(2).to_dict('index'))
