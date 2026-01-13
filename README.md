# SpikingTransformer

```bash
python train_spikeformer.py --use_wandb --wandb_project spikeformer-ocr --num_steps 4 --in_channels 1 --batch_size 72 --use_mssa --use_curriculum
```

```bash
nohup python train_spikeformer.py --use_wandb --wandb_project spikeformer-ocr --num_steps 4 --in_channels 1 --batch_size 72 --use_mssa --use_curriculum > train.logs 2>&1 &
```