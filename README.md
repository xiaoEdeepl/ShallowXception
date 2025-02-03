# DeepFake Detection Cross-Dataset Generalization Experiment

## 🎯 Experiment Objective
**Compare the generalization capability between:**
- **Original Xception** 
- **Modified ShallowXception** (with Blocks 4-11 removed)

**Training Strategy:**
- 🏋️ **Train on:** FaceForensics++ (FF++) dataset
- 🧪 **Test on:** DFDC dataset

## 📂 Dataset Architecture

### Root Directory Structure
```text
dataset/
├── dfdc/                     # DFDC test set
│   ├── video_0.mp4          # 50% real samples
│   ├── video_1.mp4          # 50% fake samples
│   └── metadata.json        # Ground truth labels
│
└── FF++/                    # FaceForensics++ training set
    ├── fake/                # 1000 manipulated videos per method
    │   ├── df/              # Deepfakes
    │   ├── f2f/             # Face2Face
    │   ├── fshift/          # FaceShift
    │   ├── fswap/           # FaceSwap 
    │   └── nt/              # NeuralTextures
    │
    └── real/                # 1000 original videos
        └── youtube/         # Sourced from YouTube
