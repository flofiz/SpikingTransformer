# SpikingTransformer

```bash
python train_spikeformer.py --use_wandb --wandb_project spikeformer-ocr --num_steps 4 --in_channels 1 --batch_size 72 --use_mssa --use_curriculum
```

```bash
nohup python train_spikeformer.py --use_wandb --wandb_project spikeformer-ocr --num_steps 4 --in_channels 1 --batch_size 72 --use_mssa --use_curriculum > train.logs 2>&1 &
```

## Multi-GPU Training (DDP)
To train on multiple GPUs (e.g., 2 GPUs), use `torchrun`:
```bash
torchrun --nproc_per_node=2 train_spikeformer.py --use_wandb --wandb_project spikeformer-ocr --num_steps 4 --in_channels 1 --batch_size 80 --use_curriculum --compile
```
```bash
nohup torchrun --nproc_per_node=2 train_spikeformer.py --use_wandb --wandb_project spikeformer-ocr --num_steps 4 --in_channels 1 --batch_size 80 --use_curriculum --compile > train.logs 2>&1 &
```
**Note:** The effective batch size will be `batch_size * nproc_per_node`. Adjust `--batch_size` accordingly (e.g. use 36 per GPU to equal 72 total).