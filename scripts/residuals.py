import json, numpy as np, pandas as pd
from servest import calib, perf
P=perf.load_params()
def table(F,Lb,M,label):
    pr=calib.predict(F,P)
    for t in calib.TARGETS:
        m=Lb[t].notna()&(Lb[t]>0)&np.isfinite(pr[t])&(pr[t]>0)
        e=np.log(pr[t][m]/Lb[t][m])
        d=M[m].copy(); d['e']=e.values
        d['conc_bin']=pd.cut(d.users.fillna(0),[-1,0,1,4,16,64,256,1e6],labels=['open','1','2-4','5-16','17-64','65-256','>256'])
        d['isl_bin']=pd.cut(d.isl,[0,512,2048,9000,40000,1e7],labels=['<512','512-2k','2k-9k','9k-40k','>40k'])
        print(f'\n=== {label} {t} n={m.sum()}  (median log ratio; + = overpredict)')
        for k in ['conc_bin','isl_bin','hw_class','engine','scheme','spec','source']:
            g=d.groupby(k,observed=True)['e'].agg(['median','count'])
            print(f'  {k:9s}', ' '.join(f'{i}:{r["median"]:+.2f}({int(r["count"])})' for i,r in g.iterrows()))
F,Lb,M=calib.load('primary'); table(F,Lb,M,'PRIMARY')
F,Lb,M=calib.load('external'); table(F,Lb,M,'EXTERNAL')
