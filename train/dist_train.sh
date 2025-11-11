#!/bin/bash

CURRENT_TIME=$(date "+%Y-%m-%d_%H:%M:%S")
echo $CURRENT_TIME
mkdir -p ./output/$CURRENT_TIME

WORLD_SIZE=8

set -u
  WORK_HOME="$PWD"
  DATA_PATH=/data/wenxing.wang/AM-DeepSeek-R1-Distilled-1.4M/am_0.9M.jsonl
  HOSTFILE=./hostfile
  LOG_FILE=./output/$CURRENT_TIME/hunyuan_large.log
  TOKENIZED_MODEL=../models
  MODEL_PATH=/data/wenxing.wang/Tencent-Hunyuan-A52B-Instruct
  SCRIPT_FILE=./train.sh
  RDZV_ID=$CURRENT_TIME
set +u

cmd="bash -c 'cd $WORK_HOME; \
     bash $SCRIPT_FILE $WORK_HOME $HOSTFILE \"$DATA_PATH\" \
     $TOKENIZED_MODEL $RDZV_ID $MODEL_PATH"

COUNT=0
hostlist=$(grep -v '^#\|^$' $HOSTFILE | awk '{print $1}' | xargs)
hostlen=$(cat $HOSTFILE | wc -l )

for host in ${hostlist[@]}; do
    ssh -p 62218 $host "pkill -f '/usr/local/bin/torchrun'" 
    echo "$host is killed."
done

COUNT=0
hostlist=$(grep -v '^#\|^$' $HOSTFILE | awk '{print $1}' | xargs)
for host in ${hostlist[@]}; do
  cmd_ssh=$cmd" > $LOG_FILE.$COUNT.$host 2>&1'"
  # cmd_ssh=$cmd" '"
  echo $cmd_ssh
  ssh -p 62218 -f -n $host $cmd_ssh
  # echo $host, "bash -c 'cd $FlagScale_HOME/megatron; nohup bash $SCRIPT_FILE $PROJ_HOME $EXPNAME $HOSTFILE \"$DATA_PATH\" >> $LOG_FILE.$COUNT.$host 2>&1 &'"
  # ssh -f -n $host "bash -c 'cd $FlagScale_HOME/megatron; nohup bash $SCRIPT_FILE $PROJ_HOME $EXPNAME $HOSTFILE \"$DATA_PATH\" >> $LOG_FILE.$COUNT.$host 2>&1 &'"
  ((COUNT++))
done

