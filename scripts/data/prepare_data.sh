#!/usr/bin/env bash
set -euo pipefail

model_path=Qwen/Qwen3-4B
config_path=config/dspark/dspark_qwen3_4b.py

dataset_name=mlabonne/open-perfectblend
local_data_dir=${LOCAL_DATA_DIR:-/mnt/data}
local_data_path=""
test_size=0.05
train_split_path=train_datasets/perfectblend_train.jsonl
eval_data_dir=eval_datasets

train_data_path=train_datasets/qwen3_4b/perfectblend_train_regen.jsonl
cache_dir=${HOME}/.cache/deepspec/qwen3_4b_target_cache

server_host=127.0.0.1
num_workers=8
start_port=30000
concurrency=32
temperature=0.7
top_p=0.8
top_k=20
min_p=0
max_tokens=4096

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

server_addresses=()
for ((worker_id = 0; worker_id < num_workers; worker_id++)); do
    server_addresses+=("${server_host}:$((start_port + worker_id))")
done

# Check if dataset already exists locally; if so, skip HuggingFace download.
# Expected layout: $local_data_dir/<dataset_name_last_segment>/
#                  e.g. /mnt/data/open-perfectblend/
dataset_local_name=$(basename "${dataset_name}")
local_candidate="${local_data_dir}/${dataset_local_name}"
if [ -d "${local_candidate}" ]; then
    local_data_path="${local_candidate}"
elif [ -f "${local_data_dir}/${dataset_local_name}.parquet" ]; then
    local_data_path="${local_data_dir}/${dataset_local_name}.parquet"
elif [ -f "${local_data_dir}/${dataset_local_name}.json" ]; then
    local_data_path="${local_data_dir}/${dataset_local_name}.json"
elif [ -f "${local_data_dir}/${dataset_local_name}.jsonl" ]; then
    local_data_path="${local_data_dir}/${dataset_local_name}.jsonl"
fi

echo "Step 1/3: splitting dataset"
if [ -n "${local_data_path}" ]; then
    echo "  Found local dataset at: ${local_data_path}"
    python scripts/data/download_and_split.py \
        --local-data-path "${local_data_path}" \
        --test-size "${test_size}" \
        --train-output-path "${train_split_path}" \
        --test-output-dir "${eval_data_dir}" \
        --skip-existing
else
    echo "  Local dataset not found in ${local_data_dir}, downloading from HuggingFace: ${dataset_name}"
    python scripts/data/download_and_split.py \
        --dataset-name "${dataset_name}" \
        --test-size "${test_size}" \
        --train-output-path "${train_split_path}" \
        --test-output-dir "${eval_data_dir}" \
        --skip-existing
fi

mkdir -p "$(dirname "${train_data_path}")"

echo "Step 2/3: generating Qwen3-4B train data: ${train_data_path}"
echo "Start sglang first with: bash scripts/data/launch_sglang_server.sh"
python scripts/data/generate_train_data.py \
    --model "${model_path}" \
    --server-address "${server_addresses[@]}" \
    --concurrency "${concurrency}" \
    --temperature "${temperature}" \
    --top-p "${top_p}" \
    --top-k "${top_k}" \
    --min-p "${min_p}" \
    --max-tokens "${max_tokens}" \
    --disable-thinking \
    --resume \
    --input-file-path "${train_split_path}" \
    --output-file-path "${train_data_path}"

echo "Stop sglang before Step 3 if it is using the same GPUs."
echo "Step 3/3: preparing Qwen3-4B target cache: ${cache_dir}"
python scripts/data/prepare_target_cache.py \
    --config "${config_path}" \
    --train-data-path "${train_data_path}" \
    --output-dir "${cache_dir}" \
    --local-batch-size 16
