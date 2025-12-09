#!/bin/bash

echo "🚀 Starting Parallel Pretraining..."
echo "  - GPU 0: Encoder (ImageNet)"
echo "  - GPU 1: Decoder (Wikipedia)"

python pretrain_encoder.py &
PID_ENC=$!

python pretrain_decoder.py &
PID_DEC=$!

echo "Processes started: Encoder ($PID_ENC), Decoder ($PID_DEC)"
wait $PID_ENC
wait $PID_DEC

echo "✅ Pretraining Completed."
