# SpikingTransformer

```bash
deepspeed --num_gpus=2 train_spikeformer.py --deepspeed --deepspeed_config ds_config.json --use_wandb --wandb_project spikeformer-ocr --num_steps 4 --in_channels 1 --batch_size 110 --use_mssa --use_curriculum --flora
```
```bash
python train_spikeformer.py --use_wandb --wandb_project spikeformer-ocr --num_steps 4 --in_channels 1 --batch_size 70 --use_mssa --use_curriculum --flora
```


```bash
torchrun --nproc_per_node=2 train_spikeformer.py --use_wandb --wandb_project spikeformer-ocr --num_steps 4 --in_channels 1 --batch_size 110 --use_mssa --use_curriculum --flora
```