# ACT-DiT 低学习率 + diffusion 目标复训 — 原始测量数据

报告：[`../../act_dit/act_dit-lowlr-diffusion-2026-08.md`](../../act_dit/act_dit-lowlr-diffusion-2026-08.md)

这里只有数据，**没有脚本**。四个脚本都在别处，且都是共用工具，不在此处分叉：

| 脚本 | 位置 |
|---|---|
| `probe_encoder_collapse.py` / `probe_conditioning.py` / `sweep_sampling.py` | [`../scripts_act_dit_probe/`](../scripts_act_dit_probe/) |
| `offline_chunk_eval.py` / `plot_horizon.py` | [`../scripts_act_eval_test/`](../scripts_act_eval_test/) |

`probe_conditioning.py` 和 `sweep_sampling.py` 在本次测量中被就地改成按 `cfg.objective`
分派（flow matching 走 Euler，diffusion 走 DDIM），细节见报告 §7。

## 文件

| 文件 | 产出脚本 | 内容 |
|---|---|---|
| `enc_lowlr.json` | `probe_encoder_collapse.py` | 7 个 checkpoint 的逐层 encoder 增益 / 幅值 / 图像敏感度 |
| `cond_lowlr_{050000,100000,150000,200000}.json` | `probe_conditioning.py` | 本次 4 个 checkpoint 的通路归因 |
| `cond_hilr_200000.json` | `probe_conditioning.py` | 08-22（diffusion, lr 1e-4）对照 |
| `lowlr_heldout.json` / `lowlr_intrain.json` | `offline_chunk_eval.py` | 本次 200k，held-out 263 ep / 训练集内 |
| `lowlr_heldout_{050000,100000,150000}.json` | `offline_chunk_eval.py` | held-out MAE 随训练步数的走向（报告 §4.1） |
| `hilr_heldout.json` / `hilr_intrain.json` | `offline_chunk_eval.py` | 08-22 对照，本次新测——分离 LR 与 objective 的唯一依据 |
| `sweep_lowlr.json` | `sweep_sampling.py` | DDIM 推理步数 × 采样平均（10/50 × 1/8） |
| `horizon_lowlr.png` | `plot_horizon.py` | 五条 held-out horizon 曲线 |

## 复现

```bash
export PYTHONPATH=/home/kewei/YING/lerobot_vlahost/src
CK=/mnt/robot_platform/jobs/act_dit_tidy_up_stationery_le_batch_success_361_2026-08-24_06-21-05-197422/run/checkpoints/200000/pretrained_model
D=/mnt/robot_platform/datasets/tidy_up_stationery_le

# 编码器体检（秒级）
conda run -n lerobot python ../scripts_act_dit_probe/probe_encoder_collapse.py --checkpoint $CK

# 通路归因（~1 min）
conda run -n lerobot python ../scripts_act_dit_probe/probe_conditioning.py \
  --checkpoint $CK --dataset-root $D/batch_success_361 --n-anchors 32

# held-out 动作误差（~90 s），--train-root 触发指纹去污染
conda run -n lerobot python ../scripts_act_eval_test/offline_chunk_eval.py --checkpoint $CK \
  --dataset-root $D/batch_1 --dataset-root $D/batch_2 --dataset-root $D/batch_3 --dataset-root $D/batch_4 \
  --train-root $D/batch_success_361 --stride 20

# 推理旋钮扫描（~15 min）
PYTHONPATH=$PYTHONPATH:../scripts_act_eval_test conda run -n lerobot python \
  ../scripts_act_dit_probe/sweep_sampling.py --checkpoint $CK \
  --dataset-root $D/batch_3 --train-root $D/batch_success_361 \
  --steps 10 --steps 50 --samples 1 --samples 8 --stride 100 --max-anchors-per-dataset 300
```

`/opt/robot-platform/train-venv`（mgmt01 那份）**不能用**：没有 EMA 代码，加载会直接失败。

## 关键读数

- `enc_lowlr.json` 的 **enc3 `image_sensitivity`**：健康 ≥1e-2（本次 2.2e-2，ACT 5.0e-2），
  塌缩 ~1e-6。一个数定生死。
- `cond_*.json` 的 **`delta_frac_of_spread.images_swapped`**：健康 >20%（本次 27.1%），
  塌缩 0%。必须固定初始噪声才有意义。
- `*_heldout.json` 的 **`aggregate.policy.mae_at_horizon["10"]` 对比 `hold_state` 的同一项**：
  前 10 帧是真正会被执行的那一段。本次是 0.0945 vs 0.0320——仍然不如原地不动。
