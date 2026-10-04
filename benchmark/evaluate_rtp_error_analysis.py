#!/usr/bin/env python
"""Locked-test error analysis for the DATA-ANCHOR RTP checkpoint.

Produces reconstruction and pgen error stratified by CDR3 length, relative
position profiles, a grouped amino-acid substitution matrix, and an empirical
BLOSUM62 comparison.  Edit distance is used for the primary residue error rate
so insertions/deletions are not silently omitted; substitution analyses retain
the explicitly labelled equal-length, aligned-position definition.
"""
from __future__ import annotations

import argparse, hashlib, json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from Bio.Align import substitution_matrices

from irrm_codec.multitask_data import (MultiTaskBenchmarkDataset, TargetStandardizer,
    build_multitask_dataloader, load_prepared_benchmark, resolve_encoder_tokenizer, select_split_indices)
from irrm_codec.multitask_transformer import IRRMCodecConfig, IRRMCodecTransformer
from irrm_codec.tokenization import decode

AA_ORDER = list("CSTAGPDEQNHRKMILVWYF")
GROUPS = ["C", "STAGP", "DEQN", "HRK", "MILV", "WYF"]
BOUNDARIES = np.cumsum([len(x) for x in GROUPS])[:-1]

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""): h.update(b)
    return h.hexdigest()

def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1): cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j-1] + (x != y)))
        prev = cur
    return prev[-1]

def args():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("data-dir", "tokenizer", "checkpoint", "output-dir"): p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=64); p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return p.parse_args()

def rate_rows(items):
    rows = []
    for length, x in sorted(items.items()):
        n=x["n"]
        rows.append({"cdr3_length": length, "n_sequences": n, "exact_sequences": x["exact"],
            "sequence_error_rate": 1-x["exact"]/n, "total_target_residues": x["res"],
            "edit_errors": x["edit"], "residue_error_rate": x["edit"]/x["res"],
            "mean_abs_log10_pgen_error": np.mean(x["pabs"]), "median_abs_log10_pgen_error": np.median(x["pabs"]),
            "mean_signed_log10_pgen_error": np.mean(x["psigned"]), "mean_true_log10_pgen": np.mean(x["truep"])})
    return pd.DataFrame(rows)

def save_figures(out, matrix, length, pos, exact):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.2, 6.8)); im=ax.imshow(matrix, cmap="Greys"); fig.colorbar(im, ax=ax, label="Wrong aligned positions")
    ax.set(xticks=range(20), xticklabels=AA_ORDER, yticks=range(20), yticklabels=AA_ORDER, xlabel="Predicted amino acid", ylabel="True amino acid", title="RTP reconstruction substitution errors")
    for x in BOUNDARIES: ax.axhline(x-.5,color="black",lw=1.25); ax.axvline(x-.5,color="black",lw=1.25)
    fig.tight_layout(); [fig.savefig((out/"figure_error_substitution_matrix").with_suffix("."+s), dpi=300 if s=="png" else None, bbox_inches="tight") for s in ("png","svg","pdf")]; plt.close(fig)
    fig, ax = plt.subplots(2,1,figsize=(8,6.4),sharex=True); x=length.cdr3_length
    ax[0].plot(x,100*length.sequence_error_rate,"o-",label="Sequence error",color="#332288"); ax[0].plot(x,100*length.residue_error_rate,"o-",label="Residue error (edit-aware)",color="#EE6677"); ax[0].set(ylabel="Error rate (%)",title="Reconstruction error increases with CDR3 length"); ax[0].legend(); ax[1].bar(x,length.n_sequences,color="#88CCEE"); ax[1].set(xlabel="CDR3 length (aa)",ylabel="Test sequences (N)")
    for a in ax: a.grid(axis="y",color="#ddd");
    fig.tight_layout(); [fig.savefig((out/"figure_error_reconstruction_by_length").with_suffix("."+s),dpi=300 if s=="png" else None,bbox_inches="tight") for s in ("png","svg","pdf")]; plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,4.5));
    for g, z in pos.groupby("length_group", sort=False): ax.plot(z.relative_bin+1,100*z.error_rate,"o-",label=g)
    ax.set(xticks=range(1,11),xlabel="Relative CDR3 position decile (N → C)",ylabel="Aligned residue error (%)",title="Central reconstruction errors persist within length strata"); ax.grid(axis="y",color="#ddd"); ax.legend(title="CDR3 length"); fig.tight_layout(); [fig.savefig((out/"figure_error_relative_position").with_suffix("."+s),dpi=300 if s=="png" else None,bbox_inches="tight") for s in ("png","svg","pdf")]; plt.close(fig)
    fig, ax = plt.subplots(2,1,figsize=(8,6.4),sharex=True); ax[0].plot(x,length.mean_abs_log10_pgen_error,"o-",color="#117733"); ax[0].set(ylabel="Mean absolute log10(pgen) error",title="pgen error by CDR3 length"); ax[1].bar(x,length.n_sequences,color="#88CCEE"); ax[1].set(xlabel="CDR3 length (aa)",ylabel="Test sequences (N)")
    for a in ax: a.grid(axis="y",color="#ddd")
    fig.tight_layout(); [fig.savefig((out/"figure_error_pgen_by_length").with_suffix("."+s),dpi=300 if s=="png" else None,bbox_inches="tight") for s in ("png","svg","pdf")]; plt.close(fig)
    fig, ax=plt.subplots(figsize=(5.5,4.3)); ax.boxplot([exact.loc[exact.exact_reconstruction,"abs_log10_pgen_error"],exact.loc[~exact.exact_reconstruction,"abs_log10_pgen_error"]],tick_labels=["Exact","Non-exact"],showfliers=False); ax.set(ylabel="Absolute log10(pgen) error",title="pgen error and reconstruction outcome"); ax.grid(axis="y",color="#ddd"); fig.tight_layout(); [fig.savefig((out/"figure_error_pgen_by_reconstruction").with_suffix("."+s),dpi=300 if s=="png" else None,bbox_inches="tight") for s in ("png","svg","pdf")]; plt.close(fig)
    # IEEE-ready single-column view: 3.5-in wide with an offset count axis.
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    plot = length.loc[length.cdr3_length.between(9, 25)].copy()
    x = plot.cdr3_length.to_numpy()
    residue_error = 100 * plot.residue_error_rate.to_numpy()
    pgen_error = plot.mean_abs_log10_pgen_error.to_numpy()
    n_sequences = plot.n_sequences.to_numpy()
    fig, ax1 = plt.subplots(figsize=(3.5, 3.15))
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.30))
    bars = ax3.bar(x, n_sequences, width=0.68, color="#BDBDBD", edgecolor="none", alpha=0.75, label="Test sequences")
    ax3.set_ylim(0, n_sequences.max() * 1.16)
    ax3.set_yticks([])
    ax3.spines["right"].set_visible(False)
    line1, = ax1.plot(x, residue_error, marker="o", markersize=3.8, linewidth=1.35, color="#0072B2", label="Residue error")
    ax1.set(xlabel="CDR3 length (aa)", ylabel="Residue (%)", xlim=(8.5, 25.5), xticks=np.arange(9, 26, 2), ylim=(0, max(15, residue_error.max() * 1.05)))
    ax1.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    ax2 = ax1.twinx()
    line2, = ax2.plot(x, pgen_error, marker="s", markersize=3.6, linewidth=1.35, linestyle="--", color="#D55E00", label=r"Mean absolute $\log_{10}(P_{gen})$ error")
    ax2.set_ylabel(r"MAE $\log_{10}(P_{gen})$", labelpad=1)
    ax2.set_ylim(0, max(0.62, pgen_error.max() * 1.05))
    for xi, ni in zip(x, n_sequences):
        if ni < 50:
            ax3.annotate(f"n={ni}", (xi, ni), textcoords="offset points", xytext=(0, 2), ha="center", fontsize=6.5, color="#4D4D4D")
    ax1.legend([bars, line1, line2], ["Test sequences", "Residue error", r"Mean $|\Delta\log_{10}P_{gen}|$"], loc="upper left", frameon=False, fontsize=7, handlelength=1.8, borderpad=0.15, labelspacing=0.25)
    for axis in (ax1, ax2, ax3):
        axis.tick_params(direction="out", length=2.5, width=0.7, labelsize=7.5)
    fig.subplots_adjust(left=0.16, right=0.82, bottom=0.18, top=0.97)
    [fig.savefig((out/"figure_error_length_combined_ieee").with_suffix("."+s), dpi=600 if s=="png" else None, bbox_inches="tight") for s in ("png", "svg", "pdf")]
    plt.close(fig)

def main():
    a=args(); out=a.output_dir
    if a.device=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    for path in (a.data_dir/"READY.json",a.data_dir/"target_standardizer.npz",a.tokenizer,a.checkpoint):
        if not path.is_file(): raise FileNotFoundError(path)
    ready=json.loads((a.data_dir/"READY.json").read_text());
    if ready.get("status")!="ready" or not all(ready.get("checks",{}).values()): raise ValueError("Benchmark is not accepting")
    payload=torch.load(a.checkpoint,map_location="cpu",weights_only=False); config=IRRMCodecConfig(**payload["model_config"])
    if payload["tokenizer"]["type"]!="data_anchor": raise ValueError("Expected DATA-ANCHOR checkpoint")
    tok=resolve_encoder_tokenizer("data_anchor",str(a.tokenizer)); table, embeddings=load_prepared_benchmark(a.data_dir); idx=select_split_indices(table,a.data_dir,"test")
    ds=MultiTaskBenchmarkDataset(table,embeddings,idx,tok,"log10_pgen_1mm",config.max_sequence_len); dev=torch.device(a.device)
    loader=build_multitask_dataloader(ds,a.batch_size,False,a.num_workers,pin_memory=dev.type=="cuda")
    model=IRRMCodecTransformer(config).to(dev); model.load_state_dict(payload["model_state"],strict=True); model.eval(); norm=TargetStandardizer.load(a.data_dir/"target_standardizer.npz")
    bylen=defaultdict(lambda:{"n":0,"exact":0,"res":0,"edit":0,"pabs":[],"psigned":[],"truep":[]}); mat=np.zeros((20,20),dtype=int); pos=defaultdict(lambda:[0,0]); region=defaultdict(lambda:[0,0]); records=[]; scores=[]; target_errors=[]; pred_errors=[]
    groups=(("≤16",lambda n:n<=16),("17–18",lambda n:17<=n<=18),("19–20",lambda n:19<=n<=20),("≥21",lambda n:n>=21))
    with torch.inference_mode():
      for batch in loader:
        t=batch["encoder_tokens"].to(dev); m=batch["encoder_mask"].to(dev); generated=[decode(r.tolist()) for r in model.reconstruct(t,m).cpu().numpy()]
        pred=(model(t,m)["pgen_standardized"].squeeze(-1).cpu().numpy()*norm.pgen_std+norm.pgen_mean)
        for target, guess, truep, pp in zip(batch["sequence"],generated,batch["pgen_target"].numpy(),pred):
          n=len(target); exact=target==guess; ed=edit_distance(target,guess); signed=float(pp-truep); item=bylen[n]; item["n"]+=1; item["exact"]+=exact; item["res"]+=n; item["edit"]+=ed; item["pabs"].append(abs(signed)); item["psigned"].append(signed); item["truep"].append(float(truep)); records.append({"cdr3_length":n,"exact_reconstruction":exact,"edit_errors":ed,"residue_error_rate":ed/n,"true_log10_pgen":float(truep),"pred_log10_pgen":float(pp),"abs_log10_pgen_error":abs(signed),"signed_log10_pgen_error":signed})
          if len(target)==len(guess):
            for j,(x,y) in enumerate(zip(target,guess)):
              if x!=y:
                mat[AA_ORDER.index(x),AA_ORDER.index(y)]+=1; scores.append(float(substitution_matrices.load("BLOSUM62")[x,y])); target_errors.append(x); pred_errors.append(y)
              for label, cond in groups:
                if cond(n):
                  key=(label,min(9,int(j*10/n))); pos[key][0]+=1; pos[key][1]+=x!=y
              r="N_edge_1_3" if j<3 else "C_edge_1_3" if j>=n-3 else "middle"; region[r][0]+=1; region[r][1]+=x!=y
    rows=[]
    for (g,b),(eligible,wrong) in pos.items(): rows.append({"length_group":g,"relative_bin":b,"eligible_positions":eligible,"wrong_positions":wrong,"error_rate":wrong/eligible})
    position=pd.DataFrame(rows).sort_values(["length_group","relative_bin"]); length=rate_rows(bylen); rec=pd.DataFrame(records); reg=pd.DataFrame([{"region":k,"eligible_positions":v[0],"wrong_positions":v[1],"error_rate":v[1]/v[0]} for k,v in region.items()])
    rng=np.random.default_rng(42); pred_counts=Counter(pred_errors); choices=np.array(list(pred_counts)); weights=np.array([pred_counts[x] for x in choices],float); weights/=weights.sum(); baseline=[]
    for x in target_errors:
      keep=choices!=x; w=weights[keep]; w/=w.sum(); baseline.append(float(substitution_matrices.load("BLOSUM62")[x,rng.choice(choices[keep],p=w)]))
    bl=pd.DataFrame({"observed_blosum62":scores,"frequency_matched_random_blosum62":baseline})
    summary={"status":"complete","analysis":"locked_test_rtp_error_analysis","test_sequences":len(rec),"checkpoint_sha256":sha256(a.checkpoint),"benchmark_ready_sha256":sha256(a.data_dir/"READY.json"),"normalizer_sha256":sha256(a.data_dir/"target_standardizer.npz"),"model_config":asdict(config),"blosum62":{"observed_mean":float(np.mean(scores)),"observed_median":float(np.median(scores)),"positive_fraction":float(np.mean(np.array(scores)>0)),"baseline_mean":float(np.mean(baseline)),"baseline_median":float(np.median(baseline)),"mean_difference_observed_minus_baseline":float(np.mean(scores)-np.mean(baseline))},"pgen_abs_error":{"exact_mean":float(rec[rec.exact_reconstruction].abs_log10_pgen_error.mean()),"nonexact_mean":float(rec[~rec.exact_reconstruction].abs_log10_pgen_error.mean()),"spearman_residue_error":float(rec.residue_error_rate.corr(rec.abs_log10_pgen_error,method="spearman")),"spearman_length":float(rec.cdr3_length.corr(rec.abs_log10_pgen_error,method="spearman"))},"note":"Substitutions and position profiles include aligned equal-length predictions only. Primary residue error is Levenshtein edit distance divided by target length."}
    out.mkdir(parents=True,exist_ok=False); pd.DataFrame(mat,index=AA_ORDER,columns=AA_ORDER).to_csv(out/"substitution_matrix.tsv",sep="\t"); length.to_csv(out/"summary_by_cdr3_length.tsv",sep="\t",index=False); position.to_csv(out/"relative_position_error_rates.tsv",sep="\t",index=False); reg.to_csv(out/"edge_middle_error_rates.tsv",sep="\t",index=False); rec.to_csv(out/"per_sequence_errors.tsv",sep="\t",index=False); bl.to_csv(out/"blosum62_observed_vs_frequency_matched_random.tsv",sep="\t",index=False); (out/"RESULTS.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    save_figures(out,mat,length,position,rec); print(json.dumps(summary,indent=2,sort_keys=True))

if __name__ == "__main__": main()
