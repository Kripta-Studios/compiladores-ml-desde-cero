"""Independent arithmetic and replay experiment, not a DeepSeek implementation."""
import json
from pathlib import Path
import numpy as np

def kv_bytes(layers, tokens, kv_heads, head_dim, bits):
    # Two distinct arrays, K and V; excludes metadata and alignment.
    return 2*layers*tokens*kv_heads*head_dim*bits//8

def causal_layers(x, layers, window):
    states=[np.array(x,dtype=np.float64)]
    for _ in range(layers):
        prev=states[-1]
        states.append(np.array([prev[max(0,i-window+1):i+1].mean()
                                for i in range(len(prev))]))
    return states[-1]

def main():
    x=np.arange(1,25,dtype=np.float64)**2
    layers,window=3,4
    full=causal_layers(x,layers,window)[-1]
    # Effective receptive field of this particular chain: 1+L*(W-1).
    exact_tokens=1+layers*(window-1)
    exact=causal_layers(x[-exact_tokens:],layers,window)[-1]
    bounded=causal_layers(x[-window:],layers,window)[-1]
    assert abs(full-exact)<1e-12
    assert abs(full-bounded)>1
    cases=[{'kv_heads':h,'bits':b,'bytes':kv_bytes(24,32768,h,64,b)}
           for h,b in [(16,16),(4,16),(4,4)]]
    assert cases[0]['bytes']==3221225472
    report={'scope':'Independent dense-KV arithmetic and causal moving-average toy; not CSA2, FP4 packing or DeepSeek execution.',
            'cache_cases':cases,'replay':{'layers':layers,'window':window,
            'exact_suffix_tokens':exact_tokens,'full_last':float(full),
            'exact_suffix_last':float(exact),'bounded_suffix_last':float(bounded),
            'bounded_absolute_error':float(abs(full-bounded))}}
    out=Path(__file__).resolve().parents[1]/'reports/research'
    out.mkdir(parents=True,exist_ok=True)
    (out/'cache_lab.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
