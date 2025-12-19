#!/bin/bash

echo "🚀 Starting Parallel Pretraining at $(date)..."
echo "  - GPU 0: Encoder (ImageNet)"
echo "  - GPU 1: Decoder (Wikipedia)"

# Lancer les processus avec nohup pour qu'ils survivent à la fermeture du terminal
nohup python pretrain_encoder.py > encoder.log 2>&1 &
PID_ENC=$!

nohup python pretrain_decoder.py > decoder.log 2>&1 &
PID_DEC=$!

echo "Processes started: Encoder ($PID_ENC), Decoder ($PID_DEC)"
echo "Log files: encoder.log, decoder. log"
echo "PIDs saved to pids.txt for monitoring"

# Sauvegarder les PIDs pour pouvoir les surveiller plus tard
echo "ENCODER_PID=$PID_ENC" > pids.txt
echo "DECODER_PID=$PID_DEC" >> pids.txt

# Attendre les deux processus
wait $PID_ENC
EXIT_ENC=$?

wait $PID_DEC
EXIT_DEC=$?

echo "✅ Pretraining Completed at $(date)"
echo "   Encoder exit code: $EXIT_ENC"
echo "   Decoder exit code: $EXIT_DEC"