# Semantics-and-Structure-Loss-for-CT-synthesis

🚩The paper **Rethinking CT Synthesis through Semantics-Structure Alignment** has been accepted by NeurIPS 2026.

## 🩻 Problem

Existing CT synthesis methods can generate visually plausible images, but they often treat all tissues uniformly and mainly optimize global pixel-, latent-, or frequency-level consistency. This can lead to two clinically important failure modes:

- **Semantic misalignment:** tissue-specific Hounsfield Unit (HU) distributions are not faithfully preserved.
- **Structural hallucination:** anatomical boundaries and unlabeled structures may be distorted, missing, or implausible.

We therefore rethink CT synthesis as a **semantics–structure alignment problem**, rather than only a visual realism problem. 

<p align="center">
  <img src="problem.png" width="90%">
</p>

## 🧠 Method

We propose **Semantics and Structure Loss (LS²)**, a plug-and-play CT-specific objective that jointly aligns **tissue-specific HU distributions** and **anatomical structures**.

LS² contains two complementary components:

- **Semantic Intensity Distribution Loss:** aligns HU distributions in clinically meaningful semantic subspaces.
- **Structural Anatomy Loss:** regularizes gradient-based anatomical structures to reduce structural hallucination.

LS² can be directly integrated into GAN, diffusion, flow-matching, and 3D foundation-model-based CT synthesis frameworks. 

<p align="center">
  <img src="framework.png" width="90%">
</p>

---

## 📈 Results

Across **GAN, diffusion, flow, and 3D foundation models**, LS² consistently improves CT synthesis in terms of **intensity fidelity, anatomical structure, semantic HU distribution, and frequency consistency**, while introducing **no additional inference overhead**.

<p align="center">
  <img src="visualization.png" width="90%">
</p>

## 💭 LS² demo

### Requirements

- Python 3.8+

```bash
pip install -r requirements.txt
```

### Data

Sample DICOM slices are included:

| File | Role |
|------|------|
| `ct_0b57b247b6_42966_00066.dcm` | Ground-truth CT (AutoPET) |
| `pred_0b57b247b6_42966_00066.dcm` | Predicted / fake CT (demo pair) |
| `AutoPET_0b57b247b6_42966_00066.dcm` | Organ / region label map |

HU values are clipped to `[-1000, 1000]` and linearly mapped to `[-1, 1]`. Images are resized to `256×256` (nearest-neighbor for the mask).

### Run

From this directory:

```bash
python demo.py
```

Custom inputs:

```bash
python demo.py \
  --real /path/to/real_ct.dcm \
  --pred /path/to/pred_ct.dcm \
  --mask /path/to/mask.dcm
```
