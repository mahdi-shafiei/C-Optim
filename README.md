# AdamW Slow, Not
Slower but Safer Update for momentum based methods
```
torchrun --standalone --nproc_per_node 1 torchrun_main.py --model_config configs/llama_60m.json --lr 0.001 --batch_size 16 --total_batch_size 512 --activation_checkpointing --num_training_steps 10000 --warmup_steps 1000 --weight_decay 0 --grad_clipping 1.0 --dtype bfloat16 --eval_every 1000 --single_gpu --optimizer adamw_slow --max_length 1024
```
