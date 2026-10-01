---
library_name: peft
tags:
- lora
- remote-sensing
- change-detection
- clip
base_model: Zilun/GeoRSCLIP
license: other
---

# LoRA adapter — GeoRSCLIP visual encoder for open-vocabulary temporal change retrieval

A LoRA adapter on the **GeoRSCLIP** visual encoder for bi-temporal, open-vocabulary change
retrieval on **Dynamic EarthNet** in NRG (near-infrared / red / green) colour mode. Part of the
GBDA lab project — full code and methodology in the
[repository](https://github.com/Panagiotis427/msc-gbda-ov-temporal-change-retrieval) and its
[technical report](https://github.com/Panagiotis427/msc-gbda-ov-temporal-change-retrieval/blob/main/report/main.pdf).

## Honest summary — a negative result

**This adapter is a research artifact, not a recommended configuration.** It **memorises the
training AOIs**: every setting of the rank/epoch sweep fits the train split (mAP 0.14–0.17) and
falls to 0.06–0.07 on the held-out test split, where the frozen zero-shot encoder reaches 0.43 (a
single 110-pair split; its cross-validated estimate is 0.100 ± 0.139).
For real retrieval, use the frozen **GeoRSCLIP + NRG** encoder with zero-shot / patch-level
Δ-scoring — the report's headline configuration (CV mAP 0.193 ± 0.051; 0.184 ± 0.046 with the corrected
class names and patch tokens, see the repository README's notes on the report). This card documents the comparison for reproducibility, not
for deployment.

## Training

- **Base model:** GeoRSCLIP (RS5M-pretrained CLIP, ViT-B/32, 512-d). Only the visual encoder is
  LoRA-adapted; the rest of the backbone stays frozen.
- **Data:** Dynamic EarthNet, `train` split, NRG colour mode; weak "X replaced by Y" change
  captions per bi-temporal pair.
- **Objective:** contrastive (masked symmetric InfoNCE) over bi-temporal change features.
- **Config / procedure:** see `src/lora_train.py` and `scripts/run_pipeline.py --lora` in the
  repository for the exact LoRA rank / alpha / epochs and training loop.

## Usage

Not a standalone model — load it through the repository:
`python -m scripts.run_pipeline … --lora`, or toggle **LoRA** in the app's Settings panel. See the
[repository README](https://github.com/Panagiotis427/msc-gbda-ov-temporal-change-retrieval).

## License

Not covered by the repository's MIT licence, which is for its code. The adapter was trained on
Dynamic EarthNet (CC BY-SA 4.0) features from GeoRSCLIP (its card says only "cc"; the RS5M data it
was trained on is CC BY-NC 4.0), so check those terms before any reuse outside research.

### Framework versions
- PEFT 0.18.1
