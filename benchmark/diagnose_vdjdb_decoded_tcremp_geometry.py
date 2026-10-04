"""Diagnose raw and target-standardized TCRemP geometry on locked VDJdb pairs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from evaluate_vdjdb_uniform100_separation import separation_metrics
from irrm_codec.multitask_data import TargetStandardizer

ROOT=Path('/home/evlasova/irrm-codec')
out=ROOT/'experiments/trb/vdjdb_decoded_tcremp/1429942/geometry_diagnostic'
out.mkdir(exist_ok=False)
cohort=pd.read_csv(ROOT/'experiments/trb/vdjdb-redcea-clonotypes/preflight-1426719/cohort/cohort.tsv',sep='\t')
pairs=np.load(ROOT/'experiments/trb/vdjdb-uniform100-v1/evaluation-1427813/uniform_pairs.npz')
original,within,between=pairs['original_indices'],pairs['within'],pairs['between']
labels=cohort.iloc[original].label.astype(str).to_numpy()
native=np.load(ROOT/'experiments/trb/vdjdb-uniform100-v1/tcremp-1429926/tcremp.npy',mmap_mode='r')
decoded=np.load(ROOT/'experiments/trb/vdjdb_decoded_tcremp/1429942/decoded_tcremp.npy',mmap_mode='r')
s=TargetStandardizer.load(ROOT/'experiments/trb/anchor-data-rtp-seed42/target_standardizer.npz')
variants={'native_raw':native[original], 'decoded_raw':decoded[original],
          'native_standardized':(native[original]-s.tcremp_mean)/s.tcremp_std,
          'decoded_standardized':(decoded[original]-s.tcremp_mean)/s.tcremp_std,
          'native_centered':native[original]-s.tcremp_mean,
          'decoded_centered':decoded[original]-s.tcremp_mean}
rows=[]
for name,matrix in variants.items():
    query,epitope=separation_metrics(matrix,within,between,labels)
    query.insert(0,'representation',name)
    query.to_parquet(out/f'per_query_{name}.parquet',index=False)
    rows.append({'representation':name,'macro_cohen_d':float(epitope.cohen_d.mean()),'macro_auroc':float(epitope.auroc.mean()),'median_cohen_d':float(epitope.cohen_d.median())})
pd.DataFrame(rows).to_csv(out/'summary.tsv',sep='\t',index=False)
(out/'RESULTS.json').write_text(json.dumps({'status':'complete','note':'All variants use unchanged separation_metrics and identical fixed uniform-100 pairs.'},indent=2)+'\n')