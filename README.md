# amsc-suf-d3-server
AmSC SUF IP Team D3 Demonstrator


## A local server

```bash
srun --job-name=TritonTest -C "gpu&hbm80g" -N 1 -G 1 -c 10 -n 1 -t 4:00:00 -A m3443 \
  -q interactive /bin/bash -c "./scripts/start-tritonserver.sh -o triton_ready.txt \
  -f cuda_models -m DoubleMetricLearning"
```