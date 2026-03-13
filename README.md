# amsc-suf-d3-server
AmSC SUF IP Team D3 Demonstrator


## Build the Docker image:
<!-- At NERSC, we use `podman-hpc`:
```bash
podman-hpc build --format docker -t docexoty/suf-d3-server .
``` -->


## Run the Triton Inference Server:
Setup CUDA models as a service.
```bash
srun --job-name=TritonGPU -C "gpu&hbm80g" -N 1 -G 1 -c 10 -n 1 -t 4:00:00 -A m3443   -q interactive /bin/bash -c "./scripts/start-tritonserver.sh -o triton_ready.txt -f cuda_models"
```

Setup CPU models as a service.
```bash
srun --job-name=TritonCPU -C "cpu" -N 1 -t 4:00:00 -A m3443 -q interactive /bin/bash -c "./scripts/start-tritonserver.sh -o triton_ready.txt -f cpu_models"
```
