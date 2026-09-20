# Training snapshot 2026-09-20 11:25:57

## GPU
index, name, utilization.gpu [%], memory.used [MiB], memory.total [MiB]
0, NVIDIA A800 80GB PCIe, 100 %, 54099 MiB, 81920 MiB
1, NVIDIA A800 80GB PCIe, 100 %, 53855 MiB, 81920 MiB

## Processes
3803437       45:23 envs/main/bin/python scripts/train/train_qlora.py --config configs/qlora_criminal.yaml --gpu 1 --micro-batch 16 --grad-accum 1 --max-batch-tokens 16384 --output-dir models/adapters/criminal --logging-dir outputs/train_logs/criminal --report-json docs/train/A0_criminal_report.json --report-md docs/train/A0_criminal_report.md
3805359       44:09 envs/main/bin/python scripts/train/train_qlora.py --config configs/qlora_criminal.yaml --gpu 1 --micro-batch 16 --grad-accum 1 --max-batch-tokens 16384 --output-dir models/adapters/criminal --logging-dir outputs/train_logs/criminal --report-json docs/train/A0_criminal_report.json --report-md docs/train/A0_criminal_report.md
3805360       44:09 envs/main/bin/python scripts/train/train_qlora.py --config configs/qlora_criminal.yaml --gpu 1 --micro-batch 16 --grad-accum 1 --max-batch-tokens 16384 --output-dir models/adapters/criminal --logging-dir outputs/train_logs/criminal --report-json docs/train/A0_criminal_report.json --report-md docs/train/A0_criminal_report.md
3819222       33:00 envs/main/bin/python scripts/train/train_qlora.py --config configs/qlora_civil.yaml --gpu 0 --micro-batch 16 --grad-accum 1 --max-batch-tokens 16384 --output-dir models/adapters/civil --logging-dir outputs/train_logs/civil --report-json docs/train/A0_civil_report.json --report-md docs/train/A0_civil_report.md
3822783       31:02 envs/main/bin/python scripts/train/train_qlora.py --config configs/qlora_civil.yaml --gpu 0 --micro-batch 16 --grad-accum 1 --max-batch-tokens 16384 --output-dir models/adapters/civil --logging-dir outputs/train_logs/civil --report-json docs/train/A0_civil_report.json --report-md docs/train/A0_civil_report.md
3822784       31:02 envs/main/bin/python scripts/train/train_qlora.py --config configs/qlora_civil.yaml --gpu 0 --micro-batch 16 --grad-accum 1 --max-batch-tokens 16384 --output-dir models/adapters/civil --logging-dir outputs/train_logs/civil --report-json docs/train/A0_civil_report.json --report-md docs/train/A0_civil_report.md

## criminal log tail

## civil log tail
