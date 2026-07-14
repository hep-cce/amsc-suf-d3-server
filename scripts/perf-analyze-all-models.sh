#!/usr/bin/env bash
set -uo pipefail

SDK_IMAGE=${SDK_IMAGE:-nvcr.io/nvidia/tritonserver:25.02-py3-sdk}
TRITON_URL=${TRITON_URL:-localhost:8001}
TRITON_HTTP_URL=${TRITON_HTTP_URL:-http://localhost:8000}
BENCHMARK_DATA=${BENCHMARK_DATA:-/global/homes/x/xju/m3443/data/AmSC_SUF_D3/BenchmarkData}
MODEL_REPOSITORY=${MODEL_REPOSITORY:-cuda_models}

curl -fsS "$TRITON_HTTP_URL/v2/health/ready" >/dev/null || {
  echo "Triton is not ready at $TRITON_HTTP_URL" >&2
  exit 1
}

passed=()
failed=()

for model_dir in "$MODEL_REPOSITORY"/*; do
  [[ -f "$model_dir/config.pbtxt" ]] || continue
  model=${model_dir##*/}
  args=(-m "$model" -u "$TRITON_URL" -i grpc --concurrency-range 1 --request-count 1)

  case "$model" in
    DoubleMetricLearning)
      args+=(--shape FEATURES:128,44)
      ;;
    nugraph2)
      args+=(--input-data /data/microbone_nugraph2_100evts.json)
      ;;
    particleNetFromMiniAODAK4CHSCentral|particleNetFromMiniAODAK4PuppiCentral)
      args+=(
        --shape pf_points:2,128 --shape pf_features:41,128 --shape pf_mask:1,128
        --shape sv_points:2,16 --shape sv_features:10,16 --shape sv_mask:1,16
        --shape lt_points:2,8 --shape lt_features:19,8 --shape lt_mask:1,8
      )
      ;;
    particleNetFromMiniAODAK4PuppiForward)
      args+=(
        --shape pf_points:2,128 --shape pf_features:40,128 --shape pf_mask:1,128
        --shape sv_points:2,16 --shape sv_features:10,16 --shape sv_mask:1,16
        --shape lt_points:2,8 --shape lt_features:19,8 --shape lt_mask:1,8
      )
      ;;
    particleNetFromMiniAODAK8)
      args+=(
        --shape pf_points:2,128 --shape pf_features:32,128 --shape pf_mask:1,128
        --shape sv_points:2,16 --shape sv_features:10,16 --shape sv_mask:1,16
      )
      ;;
    particlenet_AK8_MD-2prong_PT)
      args+=(
        --shape pf_points__0:2,128 --shape pf_features__1:20,128 --shape pf_mask__2:1,128
        --shape sv_points__3:2,16 --shape sv_features__4:11,16 --shape sv_mask__5:1,16
      )
      ;;
    particlenet_AK8_MassRegression_PT|particlenet_PT)
      args+=(
        --shape pf_points__0:2,128 --shape pf_features__1:25,128 --shape pf_mask__2:1,128
        --shape sv_points__3:2,16 --shape sv_features__4:11,16 --shape sv_mask__5:1,16
      )
      ;;
    unifiedparticletransformer_AK4_V01)
      args+=(
        --shape input_1:128,25 --shape input_2:16,18 --shape input_3:8,8 --shape input_4:8,14
        --shape input_5:128,4 --shape input_6:16,4 --shape input_7:8,4 --shape input_8:8,4
      )
      ;;
  esac

  echo "===== $model ====="
  if podman-hpc run --rm --net=host \
    -v "$BENCHMARK_DATA:/data:ro" \
    --entrypoint perf_analyzer "$SDK_IMAGE" "${args[@]}"; then
    passed+=("$model")
  else
    failed+=("$model")
  fi
done

echo
echo "PASS (${#passed[@]}): ${passed[*]}"
echo "FAIL (${#failed[@]}): ${failed[*]}"
(( ${#failed[@]} == 0 ))
