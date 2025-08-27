#!/bin/bash
# filepath: /home/yifanniu/dongyiliu/scripts/run_safety_classfier.sh

# 设置默认参数
BATCH_SIZE=10
# MODEL_PATH="/home/yifanniu/dongyiliu/bfpo/models/sft-selective"
MODEL_PATH="/data/yifanniu/Mistral-7B-v0.1"
DATA_PATH="/home/yifanniu/dongyiliu/SafetyBench/SafeRLHFfull_test_safety.json"
OUTPUT_PATH="/home/yifanniu/dongyiliu/SafetyBench/SafeRLHF_output_base.json"

KEY="prompt"
echo "Processing dataset: SafeRLHF"
# 运行 Python 脚本
python /home/yifanniu/dongyiliu/scripts/evaluation.py\
    --data_path "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_PATH" \
    --key "$KEY"

DATA_PATH="/home/yifanniu/dongyiliu/SafetyBench/AdvBenchfull_safety.json"
OUTPUT_PATH="/home/yifanniu/dongyiliu/SafetyBench/AdvBenchfull_output_base.json"
KEY="prompt"
echo "Processing dataset: Advbench"
# 运行 Python 脚本
python /home/yifanniu/dongyiliu/scripts/evaluation.py\
    --data_path "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_PATH" \
    --key "$KEY"

DATA_PATH="/home/yifanniu/dongyiliu/SafetyBench/HarmBenchfull_safety.json"
OUTPUT_PATH="/home/yifanniu/dongyiliu/SafetyBench/HarmBenchfull_output_base.json"
KEY="prompt"
echo "Processing dataset: HarmBench"
python /home/yifanniu/dongyiliu/scripts/evaluation.py\
    --data_path "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_PATH" \
    --key "$KEY"

# DATA_PATH="/home/yifanniu/dongyiliu/SafetyBench/HarmfulQAfull_safety.json"
# OUTPUT_PATH="/home/yifanniu/dongyiliu/SafetyBench/HarmfulQAfull_output.json"
# KEY="question"

# echo "Processing dataset: HarmfulQA"
# python /home/yifanniu/dongyiliu/scripts/evaluation.py\
#     --data_path "$DATA_PATH" \
#     --model_path "$MODEL_PATH" \
#     --batch_size "$BATCH_SIZE" \
#     --output_path "$OUTPUT_PATH" \
#     --key "$KEY"