# amsc-suf-d3-server
AmSC SUF IP Team D3 Demonstrator


## A local server

```bash
srun --job-name=TritonTest -C "gpu&hbm40g" -N 1 -G 1 -c 10 -n 1 -t 4:00:00 -A m3443 \
  -q interactive /bin/bash -c "./scripts/start-tritonserver.sh -o triton_ready.txt \
  -f cuda_models"
```

### A local server on CPUs
```bash
srun --job-name=TritonTest -C cpu -N 1 --exclusive -t 4:00:00 -A m3443 \
  -q interactive /bin/bash -c "./scripts/start-tritonserver.sh -o triton_ready.txt \
  -f staged_models "
```

### Start the container for debugging
```bash
podman-hpc run --rm -it --gpus all --ipc=host --net=host --pid=host \
  -v /pscratch/sd/x/xju/code/amsc-suf-d3-server:/workspace/amsc-suf-d3-server \
  -v /global/homes/x/xju/m3443/data/AmSC_SUF_D3/BenchmarkData:/global/homes/x/xju/m3443/data/AmSC_SUF_D3/BenchmarkData \
  -w /workspace/amsc-suf-d3-server \
  docker.io/docexoty/tritonserver:light

python cuda_models/nugraph2/1/model.py
```

### Create container
```bash
podman-hpc build --format docker -f Dockerfile -t docexoty/tritonserver:light
podman-hpc push localhost/docexoty/tritonserver:light docker.io/docexoty/tritonserver:light
podman-hpc pull docker.io/docexoty/tritonserver:light
```
