# DeepFake Detection Cross-Dataset Generalization Experiment

## 🎯 **Experiment Objective**
**Compare the generalization capability between:**
- **Original Xception** 
- **Modified ShallowXception** (with Blocks 4-11 removed)

**Training Strategy:**
- 🏋️ **Train on:** FaceForensics++ (FF++) dataset
- 🧪 **Test on:** DFDC dataset

## 📂 **Dataset Architecture**

### Dataset Directory Structure
```text
dataset/
├── dfdc/                    # DFDC test set
│   ├── video_0              # 14% real samples,86% fake samples
│   │   ├──frame1.jpg
│   │   ├──frame2.jpg
│   │   └──...
│   │   
│   ├── video_1              
│   │   ├──frame1.jpg
│   │   ├──frame2.jpg
│   │   └──...
│   │   
│   ├── ... 
│   └── metadata.csv         # Ground truth labels
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
```

## 🔬 **Experimental Design**

### 1. **Model Variants**
| Model          | Backbone Architecture      | Trainable Params | Pretrained |
|----------------|----------------------------|------------------|------------|
| Xception       | Original Xception          | 20.8M            | None       |
| ShallowXception| Xception (Blocks 4-11 removed) | -                | None       |

### 2. **Training Protocol**
- **Input:** 3x299x299 RGB frames (5 fps sampling from every fake video, 25 fps from every real video)
- **Optimizer:** Adam (lr=1e-4)

## 📊 **Preliminary Results (FF++ → DFDC)**

### **Cross-Dataset Performance**
| Model          | ACC on FF++ (%) | ACC on DFDC (%) |
|----------------|-----------------|-----------------|
| Xception       | **98.7**        | 51.7            |
| ShallowXception| 94              | **62.5**        |

## ⚡ **Training**

### 1. **Basic Training Command**   
```bash  
  python train.py --model Xception --epoch 20 --bs 16 --lr 1e-4
```  
- `--model`: Select model type (`Xception` or `ShallowXception`)  
- `--epoch`: Total number of epochs  
- `--bs`: Batch size  
- `--lr`: Initial learning rate  

### 2. **Arguments**  
|  Parameter |  Type |  Default | Description                                          |  
|----------------|----------|---------------|------------------------------------------------------|  
| `--c`          |    -     |     -         | Resume training from checkpoint                      |  
| `--epoch`      | int      | 10            | Number of epochs                                     |  
| `--model`      | str      | "xception"    | Model selection <br> `xception` or `shallowxception` |  
| `--bs`         | int      | 8             | Batch size                                           |  
| `--lr`         | float    | 1e-4          | Learning rate                                        |  
| `--v`          |    -     |     -         | Enable t-SNE visualization                           |
|`--dataset`     |  str     |    ff         |train dataset (ff, dfdc, cdf)                         |

### 3. **Output Files**   
The script generates the following files:  
1. **Model weights**: Saved in `./weight/` as `<model_name>.pth`  
2. **Training logs**: Saved in `./log/` as `<model_name>_logs.csv`  
3. **Loss/Accuracy plots**: Saved in `./figures/` with timestamps  
4. **t-SNE visualization (optional)**: Saved in `./figures/`  

