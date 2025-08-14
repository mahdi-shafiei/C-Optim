---

# Cautious Optimizers (C-Optiom): Improving Training with One Line of Code
AdamW has long been the go-to optimizer for transformer pretraining. For years, the research community has been searching for faster and more stable optimizers, with a focus on achieving only positive outcomes. In this work, we introduce a simple, single-line modification in PyTorch for any momentum-based optimizer. This modification, termed **Cautious Optimizer** (e.g., **C-AdamW** and **C-Lion**), opens the door to improved training performance.

Our theoretical findings reveal that this modification preserves Adam’s Hamiltonian function and retains its convergence guarantees under Lyapunov analysis. Additionally, a new family of optimizers emerges from this insight. Among these, we select the simplest for empirical experiments, achieving up to **1.47x speed-up** on **Llama** and **MAE pretraining**.

---

## 🌟 News
- **[2025-08-14]** Chinchilla Optimal (20x) runs
  - [Llama3 1B pretrained on FineWeb-edu 20x Chinchilla with AdamW](https://huggingface.co/kz919/llama3_1b_chinchilla_8132025)
  - [Llama3 1B pretrained on FineWeb-edu 20x Chinchilla with C-AdamW](https://huggingface.co/kz919/llama3_1b_cautious_chinchilla_8142025)
  
- **[2025-08-07]** [Implementing C-AdamW with parallel apply by popular demand](https://github.com/kyleliang919/C-Optim/blob/main/c_adamw.py) 🚨🚨🚨 Under current implementation with FSDP, masking and scaling would behave differently, since syncing would take took long 🚨🚨🚨       
- **[2025-01-23]** [PPO (Reinforcement Learning)](https://github.com/kyleliang919/C-Optim/blob/main/ppo_tldr.py)
- **[2025-01-14]** [Post Training experiment on Qwen2.5 1.5B Instruct](https://github.com/kyleliang919/C-Optim/blob/main/post_training.py)
- **[2024-12-03]** 🤗🤗🤗 More validation runs on ViTs [timm-optim-caution](https://huggingface.co/rwightman/timm-optim-caution)
- **[2024-12-03]** 🤗🤗🤗 Caution implemented in [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/adamw.py#L132-L136).
- **[2024-11-24]** Pre-release paper available on arXiv: [Cautious Optimizers: Improving Training with One Line of Code](https://arxiv.org/abs/2411.16085).
- **[2024-11-24]** Official implementation of C-Optim released! Experiment with C-AdamW and C-Lion today.

---

## 🚀 Implementation

### Generic Single-Line Implementation for C-Optim
<div align="center">
  <img src="images/c_optim_overview.png" alt="Image 1" style="width: 550px; margin: 0 auto;">
</div>

---

### Pretraining Results

<div align="center">
  <img src="images/c_optim_results.png" alt="Image 2" style="width: 650px; margin: 0 auto;">
</div>

---

### Post Training Results
<div align="center">
  <img src="images/c_adamw_post_training.png" alt="Image 3" style="width: 650px; margin: 0 auto;">
</div>

---

### PPO
<div align="center">
  <img src="images/ppo_tldr.png" alt="Image 3" style="width: 650px; margin: 0 auto;">
</div>

---

## 📦 Installation
### Install Experiment Dependencies
```bash
pip install -r requirements.txt
```

---

## 🛠️ Usage
### Pretraining Llama on C4
```bash
torchrun --standalone --nproc_per_node 1 torchrun_main.py \
         --model_config configs/llama_60m.json \
         --lr 0.001 \
         --batch_size 16 \
         --total_batch_size 512 \
         --activation_checkpointing \
         --num_training_steps 10000 \
         --warmup_steps 1000 \
         --weight_decay 0 \
         --grad_clipping 1.0 \
         --dtype bfloat16 \
         --eval_every 1000 \
         --single_gpu \
         --optimizer c-adamw \
         --max_length 1024
```

### Pretraining MAE on ImageNet 1K (50 Epochs)
```bash
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
### Post Training Qwen2.5
```
torchrun \
    --rdzv_id=$JOB_ID \
    --rdzv-backend=c10d \
    --nnodes=1:8 \
    --nproc-per-node=1 \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    post_training.py --model "Qwen/Qwen2.5-1.5B-Instruct" \
                     --output_dir cautious_1.5b \
                     --per_device_train_batch_size 1 \
                     --gradient_accumulation_steps 2 \
                     --max_length 8192 \
                     --cautious
```

---

### PPO
```
accelerate launch ppo_tldr.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --dataset_test_split validation \     
    --output_dir models/minimal/ppo_tldr \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --total_episodes 1000000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --local_rollout_forward_batch_size 16 \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --eval_strategy steps \
    --eval_steps 100 \
    --custom_optim c_adamw \
    --num_gpus 8
```

---

## 📖 Citation
```bibtex
@article{liang2024cautious,
  title={Cautious optimizers: Improving training with one line of code},
  author={Liang, Kaizhao and Chen, Lizhang and Liu, Bo and Liu, Qiang},
  journal={arXiv preprint arXiv:2411.16085},
  year={2024}
}
```

--- 
