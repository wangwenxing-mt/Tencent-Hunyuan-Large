#!/bin/bash

set -u
  WORK_HOME=$1
  HOSTFILE=$2
  DATA_DIR=$3
  TOKENIZED_MODEL=${4}
  RDZV_ID=${5}
  MODEL_PATH=$6
set +u

# 当前机器ip
#export LOCAL_IP=${ip1}
# 多节点机器ip，逗号隔开
#export NODE_IP_LIST="${ip1}:8,${ip2}:8"
# 机器节点个数
#export NODES=2
#export NODE_NUM=$((${NODES} * ${HOST_GPU_NUM}))

#export NCCL_DEBUG=WARN
# export TF_CPP_MIN_LOG_LEVEL=3
export DS_ACCELERATOR=musa
export LOGLEVEL="INFO"
export MUSA_EXECUTION_TIMEOUT=20000000
export MUSA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export MUSA_KERNEL_TIMEOUT=3200000
export MCCL_PROTOS=2
export MCCL_CHECK_POINTERS=0
export OMP_NUM_THREADS=4
export MCCL_ALGOS=1
# export MUDNN_LOG_LEVEL=INFO

export MCCL_BUFFSIZE=20971520
export MUSA_BLOCK_SCHEDULE_MODE=1
export MCCL_IB_GID_INDEX=3
export MCCL_NET_SHARED_BUFFERS=0
export MCCL_IB_TC=106
export MCCL_IB_QPS_PER_CONNECTION=16
export MCCL_IB_TIMEOUT=20
export MCCL_IB_RETRY_CNT=7
export MCCL_SOCKET_IFNAME=bond0
export MCCL_CROSS_NIC=0
export MUSA_LAUNCH_BLOCKING=1
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=disabled

#model_path=/data/wenxing.wang/models/241116/global_step559116
#tokenizer_path=/data/wenxing.wang/models/241116/global_step559116
#train_data_file=/data/wenxing.wang/AM-DeepSeek-R1-Distilled-1.4M/am_0.9M_sample_1k.jsonl

#ds_config_file=ds_zero2_no_offload.json
ds_config_file=ds_zero3_no_offload.json
#ds_config_file=ds_zero3_offload_no_auto.json

output_path=${WORK_HOME}/hf_train_output

mkdir -p ${output_path}

#current_time=$(date "+%Y.%m.%d-%H.%M.%S")
log_file=${output_path}/"log_${RDZV_ID}"

#echo $NODE_IP_LIST > env.txt 2>&1 &
#sed "s/:/ slots=/g" env.txt | sed "s/,/\n/g" >  "hostfile"
#sed "s/:.//g" env.txt | sed "s/,/\n/g" >  "pssh.hosts"
# export CHIEF_IP=10.10.142.120

export NODE_ADDR=$(hostname -i | tr ' ' '\n' | grep "^10\.10\.142\." | head -n1)
export GPUS_PER_NODE=8
export NUM_NODES=$(cat $HOSTFILE | wc -l)
export MASTER_ADDR=$(head -n1 $HOSTFILE | awk '{print $1;}')
export NODE_RANK=$(awk -v node_addr="$NODE_ADDR" '{ranks[$1]=(FNR-1);} END {print ranks[node_addr];}' $HOSTFILE)
export MASTER_PORT=12356

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
    --log_dir $log_file
    --redirects 3
)

echo $NODE_ADDR
echo $NUM_NODES
echo $NODE_RANK
echo $MASTER_ADDR
echo $MASTER_PORT

# HOST_PATH=./hostfile
# HOST_PATH=none

torchrun ${DISTRIBUTED_ARGS[@]} \
    train.py \
    --do_train \
    --model_name_or_path ${MODEL_PATH} \
    --tokenizer_name_or_path ${TOKENIZED_MODEL} \
    --train_data_file ${DATA_DIR} \
    --deepspeed ${ds_config_file} \
    --output_dir ${output_path}/checkpoint \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --lr_scheduler_type cosine_with_min_lr \
    --logging_steps 1 \
    --max_steps 200 \
    --save_steps 100 \
    --learning_rate 1e-5 \
    --min_lr 1e-6 \
    --warmup_ratio 0.01 \
    --save_strategy steps \
    --save_safetensors False \
    --hidden_size 6400 \
    --intermediate_size 18304 \
    --model_max_length 4096 \
    --max_seq_length 4096 \
    --moe_topk 1 \
    --num_experts 2 \
    --num_attention_heads 80 \
    --num_key_value_heads 8 \
    --num_layers 4 \
    --cla_share_factor 2 \
    --use_cla \
    --use_mixed_mlp_moe \
    --num_shared_expert 1 \
    --use_qk_norm \
    --bf16 \
    --dataloader_num_workers 4
   # --use_lora \
   # --lora_rank 64 \
   # --lora_alpha 128 \
   # --lora_dropout 0.1 \
   # --model_name_or_path ${model_path} \
