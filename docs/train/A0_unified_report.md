# QLoRA 训练报告 —— A0_unified_qlora

> 生成时间：2026-09-20 03:47:25　词表：`Qwen3-8B` + LoRA
> 结论：**PASS**

## 1. 关键检查

| 检查 | 结果 |
|---|---|
| `chat_template_renders` | ✅ |
| `assistant_only_loss` | ✅ |
| `lora_trainable_gt_0` | ✅ |
| `bucketed_sampler_active` | ✅ |
| `batch_token_budget_respected` | ✅ |
| `rendered_gt_0` | ✅ |
| `drop_rate_ok` | ✅ |
| `assistant_token_share_plausible` | ✅ |
| `loss_no_nan` | ✅ |
| `loss_decreased` | ✅ |
| `vram_peak_recorded` | ✅ |

## 2. 训练配置

| 项 | 值 |
|---|---|
| `base_model` | /mnt/data/lidian/law-agent/models/Qwen3-8B |
| `train_file` | data/train/train.jsonl |
| `dev_file` | data/dev/dev.jsonl |
| `smoke` | 0 |
| `max_seq_length` | 2048 |
| `micro_batch_size` | 4 |
| `micro_batch_size_effective` | 16 |
| `gradient_accumulation_steps` | 4 |
| `max_batch_tokens` | 16384 |
| `lr` | 0.0001 |
| `epochs` | 2.0 |
| `lora_r` | 16 |
| `lora_alpha` | 32 |
| `target_modules` | ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'] |
| `quantization` | {'load_in_4bit': True, 'bnb_4bit_quant_type': 'nf4', 'bnb_4bit_use_double_quant': True, 'bnb_4bit_compute_dtype': 'bfloat16'} |
| `enable_thinking` | False |
| `warmup_ratio_cfg` | 0.03 |
| `warmup_steps_actual` | 375 |
| `total_steps_est` | 12500 |
| LoRA 可训练参数 | **43646976** / 4761498624（0.9167%） |

## 3. 数据

| 项 | 值 |
|---|---|
| 训练样本 | 100000 |
| 扫描-rows | 100000 |
| 扫描-has_messages | 100000 |
| 扫描-single_turn | 100000 |
| 扫描-multi_turn | 0 |
| 扫描-no_assistant | 0 |
| 扫描-skipped_no_messages | 0 |
| 预渲染-scanned | 2000 |
| 预渲染-rendered | 2000 |
| 预渲染-dropped | 0 |
| 预渲染-drop_reasons | {} |
| 预渲染-truncated | 0 |
| 预渲染-asst_tok | 477887 |
| 预渲染-tot_tok | 1529968 |

> dataset_stats 由 DataLoader 子进程累加，num_workers>0 时父进程可能读到 0；**以 dataset_preview 为准**

### 运行环境适配（transformers 5.x 字段变更，已自动换算并留痕）

| 项 | 处置 |
|---|---|
| `warmup_ratio -> warmup_steps` | 0.03(ratio) x 12500(est_total_steps) -> 375 steps |
| `logging_dir` | 字段已被 transformers 5.x 删除；已手动建 outputs/train_logs/A0_unified（日志随 output_dir 落盘） |
| `steps_per_epoch_est` | 6250 |

## 4. 训练结果

| 指标 | 值 |
|---|---|
| `train_runtime` | 49549.9912 |
| `train_samples_per_second` | 4.036 |
| `train_steps_per_second` | 0.268 |
| `total_flos` | 4.5308532151641907e+18 |
| `train_loss` | 0.6756441619030572 |
| `epoch` | 2.0 |
| 峰值显存 (GB) | 46.44 |
| 训练耗时 (s) | 49550.4 |

### 长度分布与批内 padding（探针用于解释吞吐）

| 项 | 值 |
|---|---|
| 长度分桶 | True |
| 采样器 | LengthBucketedBatchSampler(bucket_width=32, batch<=16, max_batch_tokens=16384) |
| 非零样本 | 100000 |
| 零长丢弃 | 0 |
| token 长度 p10 | 120 |
| token 长度 p50 | 368 |
| token 长度 p90 | 907 |
| token 长度 p99 | 2048 |
| token 长度 max | 2048 |
| token 长度 mean | 482.3 |

## 5. loss 曲线（前 40 个记录点）

| step | loss |
|---|---|
| 10 | 2.0807 |
| 20 | 1.9039 |
| 30 | 1.8453 |
| 40 | 1.7730 |
| 50 | 1.2628 |
| 60 | 1.2919 |
| 70 | 1.2367 |
| 80 | 1.1074 |
| 90 | 1.0506 |
| 100 | 0.9412 |
| 110 | 0.9966 |
| 120 | 0.9822 |
| 130 | 0.9511 |
| 140 | 0.8879 |
| 150 | 1.0188 |
| 160 | 0.8786 |
| 170 | 0.9089 |
| 180 | 0.9283 |
| 190 | 0.9739 |
| 200 | 0.9958 |
| 210 | 0.9267 |
| 220 | 0.8900 |
| 230 | 0.8385 |
| 240 | 0.8352 |
| 250 | 0.8338 |
| 260 | 0.8771 |
| 270 | 0.9503 |
| 280 | 0.7754 |
| 290 | 0.9048 |
| 300 | 0.8249 |
| 310 | 0.8044 |
| 320 | 0.7830 |
| 330 | 0.8754 |
| 340 | 0.8689 |
| 350 | 0.8556 |
| 360 | 0.7710 |
| 370 | 0.7081 |
| 380 | 0.8531 |
| 390 | 0.9123 |
| 400 | 0.7640 |
