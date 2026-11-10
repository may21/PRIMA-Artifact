# Source data of the revised manuscript's Figures 6 and 7

Provided by the authors; these are the numbers plotted in the revised paper.
`scripts/analysis/compare_with_paper.py` uses them as the reference for
Figures 6c, 6d and 7 (instead of the values quoted in the text).

| File | Figure | Layout |
|---|---|---|
| `fig6c_page_faults_per_sec_series.csv` | 6c page-fault rate | rows CLS..OBB; 100 samples per mode, columns grouped TS, PRIMA-TS, PRIMA-MPS, MPS |
| `fig6d_gpu_utilization_series.csv` | 6d GPU utilization | one column per mode (TS, PRIMA-TS, PRIMA-MPS, MPS), one tegrastats sample per row |
| `fig7a_normalized_inference_time.csv` | 7a | mean inference time / isolated mean, per workload and MPS-2/PRIMA-2/MPS-4/PRIMA-4/MPS-8/PRIMA-8 |
| `fig7b_peak_memory_gb.csv` | 7b | peak memory (GB), same layout |
| `fig7c_page_faults_mean_sd.csv` | 7c | page faults/s, two columns per configuration: mean, SD |

Known disagreements with the manuscript text (reported by the comparison tool):
Figure 6d medians are 0/56/0/65 % (TS/PRIMA-TS/MPS/PRIMA-MPS) while the text
states 49→98 % and 46→97 %; Figure 6c has PRIMA-MPS above MPS for every
workload (OBB +61 %) while the text states a 37.9 % reduction.
