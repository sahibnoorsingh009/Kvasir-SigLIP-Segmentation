# Prespecified methodology

## Primary question

Does SigLIP2 improve foreground polyp segmentation over a reproduced ResUNet baseline
on the same author-released Kvasir-SEG evaluation images?

## Primary estimand

For each official evaluation image i:

d_i = Dice_i(SigLIP2) - Dice_i(ResUNet)

Primary effect = mean_i(d_i).

## Decision rule

- Statistical evidence of improvement: paired bootstrap 95% CI lies entirely above 0.
- Prespecified practical improvement: lower CI bound is at least +0.02 Dice.
- Inconclusive: CI crosses 0.
- Evidence of worse performance: CI lies entirely below 0.

## Protocol

1. Download Kvasir-SEG files from Kaggle.
2. Download the authors' split files from GitHub.
3. Keep official `val.txt` untouched as final benchmark data.
4. Split official `train.txt` once into internal train and internal validation.
5. Reproduce five-stage ResUNet using the paper recipe.
6. Train SigLIP2-UNet with frozen, partial, and full encoder modes.
7. Select modes and checkpoints using internal validation only.
8. Evaluate finalized methods once on official evaluation data.
9. Run three seeds for each finalized method.
10. Report point estimates, seed variability, paired bootstrap CIs, and permutation p-values.

## Important limitations

- The repository split is author-released but differs from the 80/10/10 split described in
  the original paper.
- Kvasir-SEG may lack patient/procedure identifiers; independence between still images cannot
  be guaranteed.
- The ResUNet and SigLIP2 decoders are not identical architectures. The main scientific
  intervention is replacing the CNN representation with a pretrained transformer representation,
  while maintaining the same data, split, loss, resolution, threshold, and evaluation.
- A later secondary experiment can use a common FPN decoder for both a pretrained ResNet and
  SigLIP2 to isolate encoder effects more strictly.
