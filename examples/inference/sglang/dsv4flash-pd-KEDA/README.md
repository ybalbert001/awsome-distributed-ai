<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# DeepSeek-V4-Flash PD with KEDA autoscaling (B200, EKS)

Prefill/decode disaggregation for DeepSeek-V4-Flash on `p6-b200.48xlarge`, with prefill and decode scaled independently by [KEDA](https://keda.sh) on latency and queue metrics read from Amazon Managed Service for Prometheus (AMP).

- Each engine is one pod (`tp=2`, 2 GPUs, 2 EFA interfaces). Four engine pods fill one 8-GPU node.
- KV moves from prefill to decode over NIXL `LIBFABRIC` (GPUDirect RDMA over EFA), on the same node or across nodes.
- An SGLang router discovers prefill and decode pods through the Kubernetes API and follows KEDA's replica changes, so scaling a Deployment is the whole scale operation.

This sample extends [`dsv41flash-pd-autodiscovery`](../dsv41flash-pd-autodiscovery), the closest existing sample (same one-engine-per-pod layout and router discovery, for DeepSeek-V4.1-Flash). It adds the autoscaling layer and uses the V4-Flash engine flags. That sample has no autoscaler, and it uses a different model and engine flags.

## How scaling works

KEDA turns the triggers below into an HPA per Deployment. The HPA takes the highest replica count over a role's triggers.

| Role | Trigger | Query (1-minute window) | Threshold |
| --- | --- | --- | --- |
| prefill | `ttft_p90` | P90 of `sglang:time_to_first_token_seconds` on the **decode** engines | 3 s |
| prefill | `prefill_queue` | `avg_over_time(sum(sglang:num_queue_reqs{component="prefill"})[1m:10s])` | 24 per replica |
| decode | `tpot_p90` | P90 of `sglang:inter_token_latency_seconds` on the decode engines | 0.015 s |
| decode | `decode_prealloc_queue` | `sum(sglang:num_decode_prealloc_queue_reqs{component="decode"})` | 8 per replica |

- TTFT is measured on the decode engines. In PD mode that is the only place a request's first token is timed end to end, and prefill engines export no TTFT.
- The queue triggers use KEDA's default `AverageValue`: desired replicas = `ceil(total / threshold)`. The latency triggers use `Value`: desired = `ceil(current replicas * value / threshold)`.
- The queue signals tell the two roles apart. TTFT also rises when decode is out of KV (prefilled requests wait for decode pre-allocation), and adding prefill would not help.
- `sglang:inter_token_latency_seconds` does not exist on an engine until it has served streaming requests, so the `tpot_p90` trigger sees no data before that and, with `ignoreNullValues: "true"`, counts as 0. The `bench_serving` load below streams, and it is enough to create the metric.
- Scale-up adds one pod every 120 s. Scale-down waits 600 s of stability and removes one pod every 300 s. A new engine needs several minutes to become Ready.
- Scale-in is graceful. A `preStop` hook waits until the engine has no running, queued or in-flight requests (up to 580 s) before SIGTERM, and `terminationGracePeriodSeconds` is 600.
- The thresholds were set from a fixed-replica 1P1D sweep (1024 input / 256 output tokens). Treat them as a starting point for your own traffic.
- `maxReplicaCount` is 2 for prefill (two-node demo) and 4 for decode. Raise it with GPU capacity.

If AMP cannot be queried, KEDA reports an error and the HPA keeps the current replica count. The ScaledObjects set no `fallback`.

## Prerequisites

- EKS or HyperPod EKS 1.33+ with `p6-b200.48xlarge` nodes, the NVIDIA device plugin and the EFA device plugin. `kubectl describe node` must show `nvidia.com/gpu: 8` and `vpc.amazonaws.com/efa: 8`.
- The EKS Pod Identity Agent add-on, for KEDA to query AMP.
- The metrics pipeline from [`observability/prometheus-grafana/eks-managed-observability`](../../../../observability/prometheus-grafana/eks-managed-observability), deployed with the **SGLang** scrape job enabled. It creates the AMP workspace and runs an ADOT collector that scrapes the engines. To also chart EFA/RDMA throughput, enable the **EFA exporter** job there as well (see that README).
- Local NVMe with room for the weights on every GPU node. HyperPod: `/opt/dlami/nvme`. Self-managed EKS: `/mnt/k8s-disks/0`.
- `kubectl`, `helm`, `envsubst`, Docker and the AWS CLI.

Tested on EKS 1.35, `p6-b200.48xlarge` x2, KEDA 2.21.0 (chart `keda-2.21.0`), router `lmsysorg/sgl-model-gateway:v0.3.2`. The engine image reported `sglang.__version__` = `0.0.0.dev1+g65f759144`.

## Setup

```bash
cd examples/inference/sglang/dsv4flash-pd-KEDA
export AWS_REGION=<eks-region>
export CLUSTER_NAME=<eks-cluster-name>
export NVME_HOST_PATH=/opt/dlami/nvme   # /mnt/k8s-disks/0 on self-managed EKS
```

### 1. Image and weights

Build the EFA-enabled engine image and pre-stage the weights on every GPU node, using the shared helpers one level up.

```bash
# Prints the ECR URI on the last line. Set BASE_IMAGE to the SGLang image that
# serves DeepSeek-V4-Flash with the flags in manifests/prefill.yaml, pinned by digest.
BASE_IMAGE=<sglang-base-image@sha256:...> ALGORITHM_NAME=sgl-dsv4-efa ../build-image.sh
export SGLANG_IMAGE=<the ECR URI printed above>

../download-model.sh deepseek-ai/DeepSeek-V4-Flash p6-b200.48xlarge
# Wait until every pod logs "Download complete!", then:
kubectl delete daemonset model-downloader
```

### 2. Deploy the engines and the router

```bash
kubectl create namespace dsv4flash
cat manifests/*.yaml | envsubst '${SGLANG_IMAGE} ${NVME_HOST_PATH}' | kubectl apply -n dsv4flash -f -
kubectl rollout status -n dsv4flash deploy/dsv4flash-prefill deploy/dsv4flash-decode deploy/dsv4flash-router --timeout=30m
```

A cold start (weight load, JIT, CUDA graphs) takes several minutes. The router becomes Ready once it has one prefill and one decode worker.

### 3. Install KEDA

```bash
helm repo add kedacore https://kedacore.github.io/charts && helm repo update
helm install keda kedacore/keda --version 2.21.0 -n keda --create-namespace \
  -f autoscaling/keda-values.yaml
```

### 4. Let KEDA read AMP

KEDA's operator queries AMP with SigV4 through EKS Pod Identity. Create a read-only role for it:

```bash
export AMP_REGION=<amp-region>
export AMP_WORKSPACE_ID=<ws-...>        # from the observability stack outputs
export AMP_QUERY_URL=https://aps-workspaces.${AMP_REGION}.amazonaws.com/workspaces/${AMP_WORKSPACE_ID}
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

aws iam create-role --role-name keda-operator-amp-query-role \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"pods.eks.amazonaws.com"},"Action":["sts:AssumeRole","sts:TagSession"]}]}'
aws iam put-role-policy --role-name keda-operator-amp-query-role --policy-name amp-query-readonly \
  --policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["aps:QueryMetrics","aps:GetLabels","aps:GetSeries","aps:GetMetricMetadata"],"Resource":"arn:aws:aps:'${AMP_REGION}':'${ACCOUNT_ID}':workspace/'${AMP_WORKSPACE_ID}'"}]}'

aws eks create-pod-identity-association --cluster-name "$CLUSTER_NAME" --region "$AWS_REGION" \
  --namespace keda --service-account keda-operator \
  --role-arn arn:aws:iam::${ACCOUNT_ID}:role/keda-operator-amp-query-role

# Pod Identity credentials are injected when a pod is created, so restart the
# operator after the association exists.
sleep 15
kubectl rollout restart -n keda deploy/keda-operator
kubectl rollout status -n keda deploy/keda-operator
kubectl get pod -n keda -l app=keda-operator -o yaml | grep AWS_CONTAINER_CREDENTIALS_FULL_URI
```

The last command must print a match. If it prints nothing, restart the operator again.

### 5. Apply the ScaledObjects

```bash
envsubst '${AMP_REGION} ${AMP_QUERY_URL}' < autoscaling/scaledobjects.yaml | kubectl apply -n dsv4flash -f -
kubectl get scaledobject,hpa -n dsv4flash
```

Both ScaledObjects should show `READY=True`. With no traffic the prefill HPA shows `0/3` for `ttft_p90` and `0/24 (avg)` for `prefill_queue`.

## Test autoscaling

The load generator is `sglang.bench_serving`, which ships in the engine image. Run it in a running engine pod against the router service. It needs no GPU and the CPU cost is small next to the engine.

### Check the signals first

Send a small streaming load and confirm the four metrics have values in AMP.

```bash
POD=$(kubectl get pod -n dsv4flash -l component=prefill -o name | head -1)
kubectl exec -n dsv4flash $POD -- python3 -m sglang.bench_serving \
  --backend sglang-oai --base-url http://dsv4flash-router:30080 \
  --model deepseek-ai/DeepSeek-V4-Flash --tokenizer deepseek-ai/DeepSeek-V4-Flash \
  --dataset-name random-ids --random-input-len 1024 --random-output-len 256 --random-range-ratio 1.0 \
  --num-prompts 16 --max-concurrency 4 --warmup-requests 0 --disable-tqdm
```

`bench_serving` prints `Request throughput`, `P90 TTFT` and `P90 TPOT`. Then query AMP (Grafana Explore, or `awscurl`) for the trigger queries in the table above, with `namespace="dsv4flash"`. In the SGLang PD dashboard from the observability stack, the TTFT and TPOT panels should fill in.

### Step load

Run time-boxed stages of rising concurrency. Each batch uses a fresh seed so the prefix cache does not flatter the numbers. `STAGES` is `concurrency:seconds` pairs.

```bash
cat > /tmp/step-load.sh <<'EOF'
set -uo pipefail
STAGES=${STAGES:-"16:150 64:150 128:150 256:180 128:150 32:120"}
IN=${IN:-1024}; OUT=${OUT:-256}
URL=http://dsv4flash-router:30080; MODEL=deepseek-ai/DeepSeek-V4-Flash
DIR=/tmp/load; mkdir -p "$DIR"; seed=$(date +%s)
for st in $STAGES; do
  c=${st%%:*}; secs=${st##*:}; end=$(( $(date +%s) + secs )); b=0
  echo "--- c=$c for ${secs}s from $(date -u +%T)"
  while [ "$(date +%s)" -lt "$end" ]; do
    b=$((b+1)); seed=$((seed+1))
    python3 -m sglang.bench_serving --backend sglang-oai --base-url "$URL" \
      --model "$MODEL" --tokenizer "$MODEL" --dataset-name random-ids \
      --random-input-len "$IN" --random-output-len "$OUT" --random-range-ratio 1.0 \
      --num-prompts $((c*4)) --max-concurrency "$c" --warmup-requests 0 \
      --seed "$seed" --disable-tqdm > "$DIR/c${c}_b${b}.log" 2>&1 \
      || { echo "batch failed: $DIR/c${c}_b${b}.log"; continue; }
    awk -F: '/Successful requests/{s=$2} /Request throughput/{r=$2} /P90 TTFT/{t=$2} /P90 TPOT/{p=$2}
      END{gsub(/ /,"",s);gsub(/ /,"",r);gsub(/ /,"",t);gsub(/ /,"",p);
          printf "ok=%s req/s=%s ttft_p90=%sms tpot_p90=%sms\n",s,r,t,p}' "$DIR/c${c}_b${b}.log"
  done
done
EOF
POD=$(kubectl get pod -n dsv4flash -l component=prefill -o name | head -1)
kubectl exec -i -n dsv4flash $POD -- bash -s < /tmp/step-load.sh
```

The default stages take about 15 minutes. The load stops when the `kubectl exec` session ends, so run it in `tmux` or `screen`.

### Watch scaling

In another terminal:

```bash
kubectl get hpa -n dsv4flash -w
kubectl get deploy,pod -n dsv4flash -l app=dsv4flash
kubectl describe hpa -n dsv4flash keda-hpa-dsv4flash-prefill   # current vs target per trigger, and scale events
```

Expect, in order:

1. At rising concurrency the prefill queue and TTFT P90 climb past their thresholds. The HPA raises prefill by one pod per 120 s, up to `maxReplicaCount`.
2. The new pod is `Running` but `0/1` until the model has loaded. Until it is Ready the router does not use it, and the dashboard's "up" replica count (`up{job="sglang"} == 1`) stays unchanged.
3. When the new prefill is Ready, TTFT P90 and the queue should fall.
4. If the decode TPOT or pre-allocation queue crosses its threshold, decode scales the same way.
5. After the load ends, replicas stay up for at least 600 s, then fall one pod per 300 s toward `minReplicaCount`.

### Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| HPA shows `<unknown>` or ScaledObject `READY=False` | KEDA cannot query AMP. Check `kubectl logs -n keda deploy/keda-operator`, and that the operator pod has `AWS_CONTAINER_CREDENTIALS_FULL_URI` (step 4). |
| HPA stays at 0 under load | The query returns no data. Run the trigger query in AMP. The engine pods need the `sglang-metrics: "true"` label and the `prometheus.io/*` annotations, and the ADOT collector needs the SGLang scrape job. |
| TPOT trigger always 0 | No streaming requests yet, so `inter_token_latency_seconds` does not exist. Run the signal check above. |
| Prefill replicas never go above 1 | `maxReplicaCount` is reached, or the new pod is still loading the model, or no node has 2 free GPUs and 2 EFA interfaces. |
| New engine `Pending` | Not enough free GPUs or EFA on the matching nodes. |
| `FailedGetExternalMetric` with 403 | The role or Pod Identity association is missing, or the operator was not restarted after creating it. |

## Cleanup

```bash
envsubst '${AMP_REGION} ${AMP_QUERY_URL}' < autoscaling/scaledobjects.yaml | kubectl delete -n dsv4flash -f -
cat manifests/*.yaml | envsubst '${SGLANG_IMAGE} ${NVME_HOST_PATH}' | kubectl delete -n dsv4flash -f -
helm uninstall keda -n keda
aws eks list-pod-identity-associations --cluster-name "$CLUSTER_NAME" --region "$AWS_REGION" --namespace keda
# delete the association it lists, then:
aws iam delete-role-policy --role-name keda-operator-amp-query-role --policy-name amp-query-readonly
aws iam delete-role --role-name keda-operator-amp-query-role
```

## Known issues

- The router's own metrics (`:29000`) are not scraped by default. The router pod has no `prometheus.io/*` annotations, so the observability stack's SGLang job skips it. KEDA does not use them.
- The ScaledObjects set no `fallback`, so an AMP outage freezes the replica count.
- The preStop drain is bounded by `terminationGracePeriodSeconds` (600 s). A request that decodes longer than that is cut off.
