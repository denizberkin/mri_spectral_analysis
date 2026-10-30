# mri_spectral_analysis

Code for team **inzva mri**'s MRIxFields 2026 Task 3 entry (cross-field brain MRI translation,
3rd Place Award) and for the spectral analysis of its data and errors.

- **Model:** 2D conditional U-Net, 4-channel multi-contrast input, source/target field
  embeddings with FiLM in the decoder, zero-init residual head, axial-position token.
- **Training:** masked-image pretraining on the unpaired retrospective split, then two
  supervised fine-tunes on the paired prospective subjects. The final weights are the mean of
  stage-2 epochs 6-8, and inference uses 4-flip test-time augmentation.
- **Spectral analysis:** radially averaged power spectra and power-law fits of the dataset,
  and the frequency and spatial distribution of the final model's error.

## Layout

| path | what |
|---|---|
| `experiment-pipeline/eval_pipeline/` | config-driven experiment runner (MIT, see `experiment-pipeline/LICENSE`) |
| `experiment-pipeline/components/` | Task 3 data module, `ConditionalUNet`, losses, trainer |
| `experiment-pipeline/configs/task3_retro_pretrain_big.toml` | stage 1: 30k-step masked-image pretraining, then 25 supervised epochs |
| `experiment-pipeline/configs/task3_mc_ssim_slice.toml` | stage 2: 8 epochs, 4-channel input, SSIM term, axial-position token |
| `mrixfields/` | dataset classes, transforms, LPIPS loss, `.env` handling |
| `scripts/preprocess.py` | NIfTI volumes to cached axial slices (72-291) |
| `scripts/widen_input_channels.py` | stage-1 checkpoint, first convolution 1 to 4 channels, new channels zero |
| `scripts/make_soup.py` | uniform weight average of checkpoints |
| `scripts/make_task3_submission.py` | inference: validation-phase submission ZIP, optional 4-flip TTA |
| `scripts/spectral_analysis.py`, `scripts/spectral_figures.py` | dataset spectra, power-law exponents, figures |
| `scripts/error_spectrum_analysis.py` | error spectrum and 1 - SSIM by edge strength and region for the final model |

## Setup

- Python 3.13, an NVIDIA GPU.
- `pip install -r requirements.txt` (install `torch` from the CUDA index that matches your driver).
- `cp .env.example .env` and set `DATA_DIR`, `PREPROCESSED_DIR`, `SUBMISSION_DIR`.

## Training the final model

Pipeline commands run from inside `experiment-pipeline/`; checkpoints land in
`experiment-pipeline/runs/<experiment>/artifacts/`.

```bash
# 1. cache axial slices of the paired and retrospective training splits
python scripts/preprocess.py extract-slices --splits pro_train retro_train

cd experiment-pipeline

# 2. stage 1: masked-image pretraining + 25 supervised epochs
python -m eval_pipeline run configs/task3_retro_pretrain_big.toml --stages training

# 3. widen the epoch-25 checkpoint to the 4-channel input
python ../scripts/widen_input_channels.py --channels 4 \
    --checkpoint runs/task3_retro_pretrain_big/artifacts/task3_unet_finetune_25.pt \
    --output     runs/task3_retro_pretrain_big/artifacts/widened4_from_e25.pt

# 4. stage 2: 8 epochs from the widened checkpoint
python -m eval_pipeline run configs/task3_mc_ssim_slice.toml --stages training

# 5. final weights: uniform mean of stage-2 epochs 6, 7, 8
python ../scripts/make_soup.py \
    --member runs/task3_mc_ssim_slice/artifacts/task3_unet_finetune_6.pt:1 \
    --member runs/task3_mc_ssim_slice/artifacts/task3_unet_finetune_7.pt:1 \
    --member runs/task3_mc_ssim_slice/artifacts/task3_unet_finetune_8.pt:1 \
    --out runs/task3_mc_ssim_slice/artifacts/avg_e6_e8.pt

cd ..

# 6. validation-phase submission with 4-flip TTA
python scripts/make_task3_submission.py --tta \
    --checkpoint experiment-pipeline/runs/task3_mc_ssim_slice/artifacts/avg_e6_e8.pt
```

## Spectral analysis

```bash
# dataset spectra and power-law exponents per (split, contrast, field)
python scripts/spectral_analysis.py --splits val retro --out-dir reports/spectral
python scripts/spectral_figures.py --in-dir reports/spectral --out-dir reports/spectral

# error of the final model on the paired subjects (reads reports/spectral/alpha_summary.csv)
python scripts/error_spectrum_analysis.py \
    --checkpoint experiment-pipeline/runs/task3_mc_ssim_slice/artifacts/avg_e6_e8.pt
```
