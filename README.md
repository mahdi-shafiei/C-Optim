# Cautious Optimizer (C-Optiom)
Cautious Update for momentum based methods
```
torchrun --standalone --nproc_per_node 1 torchrun_main.py --model_config configs/llama_60m.json --lr 0.001 --batch_size 16 --total_batch_size 512 --activation_checkpointing --num_training_steps 10000 --warmup_steps 1000 --weight_decay 0 --grad_clipping 1.0 --dtype bfloat16 --eval_every 1000 --single_gpu --optimizer c-adamw --max_length 1024
```


```
torchrun --standalone --nproc_per_node 4 run_mae.py \
    --dataset_name ILSVRC/imagenet-1k \
    --output_dir ./vit-mae \
    --remove_unused_columns False \
    --label_names pixel_values \
    --mask_ratio 0.75 \
    --norm_pix_loss \
    --do_train \
    --do_eval \
    --base_learning_rate 1.5e-4 \
    --lr_scheduler_type cosine \
    --weight_decay 0.05 \
    --num_train_epochs 50 \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337 \
    --custom_optim adamw_torch \
    --trust_remote_code \
    --gradient_accumulation_steps 4
```

```
torchrun --standalone --nproc_per_node 4 run_mae.py \
    --dataset_name ILSVRC/imagenet-1k \
    --output_dir ./vit-mae-c \
    --remove_unused_columns False \
    --label_names pixel_values \
    --mask_ratio 0.75 \
    --norm_pix_loss \
    --do_train \
    --do_eval \
    --base_learning_rate 1.5e-4 \
    --lr_scheduler_type cosine \
    --weight_decay 0.05 \
    --num_train_epochs 50 \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337 \
    --custom_optim c-adamw \
    --trust_remote_code \
    --gradient_accumulation_steps 4
```