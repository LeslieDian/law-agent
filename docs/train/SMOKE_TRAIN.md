# QLoRA 训练报告 —— A0_unified_qlora_SMOKE

> 生成时间：2026-09-19 13:29:56　词表：`Qwen3-8B` + LoRA
> 结论：**FAIL**

## 1. 关键检查

| 检查 | 结果 |
|---|---|
| `chat_template_renders` | ✅ |
| `assistant_only_loss` | ✅ |
| `lora_trainable_gt_0` | ✅ |
| `rendered_gt_0` | ❌ |
| `drop_rate_ok` | ✅ |
| `loss_no_nan` | ✅ |
| `loss_decreased` | ✅ |
| `vram_peak_recorded` | ✅ |

## 2. 训练配置

| 项 | 值 |
|---|---|
| `base_model` | /mnt/data/lidian/law-agent/models/Qwen3-8B |
| `train_file` | data/train/train.jsonl |
| `dev_file` | data/dev/dev.jsonl |
| `smoke` | 500 |
| `max_seq_length` | 4096 |
| `micro_batch_size` | 4 |
| `gradient_accumulation_steps` | 4 |
| `lr` | 0.0001 |
| `epochs` | 1 |
| `lora_r` | 16 |
| `lora_alpha` | 32 |
| `target_modules` | ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'] |
| `quantization` | {'load_in_4bit': True, 'bnb_4bit_quant_type': 'nf4', 'bnb_4bit_use_double_quant': True, 'bnb_4bit_compute_dtype': 'bfloat16'} |
| `enable_thinking` | False |
| LoRA 可训练参数 | **43646976** / 4761498624（0.9167%） |

## 3. 数据

| 项 | 值 |
|---|---|
| 训练样本 | 500 |
| 扫描-rows | 500 |
| 扫描-has_messages | 500 |
| 扫描-single_turn | 500 |
| 扫描-multi_turn | 0 |
| 扫描-no_assistant | 0 |
| 扫描-skipped_no_messages | 0 |
| 数据集-rendered | 0 |
| 数据集-dropped | 0 |
| 数据集-drop_reasons | {} |
| 数据集-truncated | 0 |
| 数据集-asst_tok | 0 |
| 数据集-tot_tok | 0 |

## 4. 训练结果

| 指标 | 值 |
|---|---|
| `train_runtime` | 196.2366 |
| `train_samples_per_second` | 1.631 |
| `train_steps_per_second` | 0.102 |
| `total_flos` | 1.427954059186176e+16 |
| `train_loss` | 0.6846774727106094 |
| `epoch` | 0.64 |
| 峰值显存 (GB) | 24.42 |
| 训练耗时 (s) | 196.6 |

## 5. loss 曲线（前 40 个记录点）

| step | loss |
|---|---|
| 1 | 1.1423 |
| 2 | 0.9851 |
| 3 | 0.8481 |
| 4 | 0.7696 |
| 5 | 0.7313 |
| 6 | 0.6794 |
| 7 | 0.6671 |
| 8 | 0.6679 |
| 9 | 0.6438 |
| 10 | 0.6605 |
| 11 | 0.6769 |
| 12 | 0.5547 |
| 13 | 0.5769 |
| 14 | 0.6494 |
| 15 | 0.6118 |
| 16 | 0.6205 |
| 17 | 0.5948 |
| 18 | 0.5353 |
| 19 | 0.5131 |
| 20 | 0.5650 |
