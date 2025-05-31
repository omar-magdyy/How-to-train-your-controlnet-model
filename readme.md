# ControlNet Training Tutorial for Windows

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive guide for training ControlNet models on Windows, with special focus on resolving the infamous **bitsandbytes** compatibility issues.

## 🚀 Quick Start

For Windows users who want to start training immediately:

```bash
# 1. Create environment
py -3.9 -m venv venv39
.\venv39\Scripts\Activate.ps1

# 2. Install dependencies (see full list below)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers accelerate transformers

# 3. Train (without bitsandbytes complications)
accelerate launch train_controlnet.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --output_dir="control-model/" \
  --train_data_dir="YOUR_DATASET_PATH" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --learning_rate=1e-5 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --num_train_epochs=4 \
  --checkpointing_steps=5000
```

## 📋 Table of Contents

- [Environment Setup](#-environment-setup)
- [Required Dependencies](#-required-dependencies)
- [Windows-Specific: bitsandbytes Issues](#-windows-specific-bitsandbytes-issues)
- [Dataset Preparation](#-dataset-preparation)
- [Metadata Generation Script](#-metadata-generation-script)
- [Training Configuration](#-training-configuration)
- [Running Training](#-running-training)
- [Troubleshooting](#-troubleshooting)
- [Hardware Recommendations](#-hardware-recommendations)
- [Post-Training](#-post-training)
- [Contributing](#-contributing)

## 🛠️ Environment Setup

### Step 1: Create Virtual Environment

```bash
# Create Python 3.9 virtual environment
py -3.9 -m venv venv39

# Activate (Windows PowerShell)
.\venv39\Scripts\Activate.ps1

# Activate (Command Prompt)
.\venv39\Scripts\activate.bat

# Verify installation
python --version  # Should show Python 3.9.x
```

## 📦 Required Dependencies

### Step 2: Install Dependencies in Correct Order

```bash
# Core ML framework
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# Fix aiohttp compatibility with Python 3.9
pip install "aiohttp<3.9.0"

# Training libraries
pip install accelerate datasets transformers diffusers

# Image processing
pip install pillow opencv-python

# Memory optimization
pip install xformers==0.0.23.post1 --extra-index-url https://pypi.nvidia.com

# Optional: Experiment tracking
pip install wandb

# Utilities
pip install numpy tqdm
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
```

## ⚠️ Windows-Specific: bitsandbytes Issues

### The Problem

Windows users often encounter this error when using `--use_8bit_adam`:

```
RuntimeError: CUDA Setup failed despite GPU being available
libcudart.so not found in any environmental path
argument of type 'WindowsPath' is not iterable
```

### Solutions (Choose One)

#### 🥇 Solution 1: Skip bitsandbytes (Recommended)

**Easiest and most reliable approach:**

- Remove `--use_8bit_adam` from training commands
- Use regular Adam optimizer instead
- Works immediately with no setup required

#### 🥈 Solution 2: Install Windows-Compatible bitsandbytes

```bash
# Method A
pip install bitsandbytes-windows

# Method B (if Method A fails)
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl
```

**Test installation:**
```bash
python -c "import bitsandbytes as bnb; print('Success!')"
```

#### 🥉 Solution 3: Fix CUDA Environment (Advanced)

1. Find CUDA installation: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\`
2. Add environment variables:
   - `CUDA_PATH`: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`
   - Add to `PATH`: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`
3. Restart command prompt and install: `pip install bitsandbytes`

## 📁 Dataset Preparation

### Dataset Structure

```
dataset/
├── images/
│   ├── 001.png
│   ├── 002.png
│   └── ...
├── conditioning_images/
│   ├── 001.png  # Control images (Canny, depth, etc.)
│   ├── 002.png
│   └── ...
├── text/
│   ├── 001.txt  # Text descriptions
│   ├── 002.txt
│   └── ...
└── metadata.jsonl  # Generated automatically
```

## 🔧 Metadata Generation Script

Save this as `generate_metadata.py`:

```python
import os
import json

# 🔧 Configure your directories
images_dir = r"path/to/your/dataset/images"
conditioning_dir = r"path/to/your/dataset/conditioning_images"
texts_dir = r"path/to/your/dataset/text"
output_path = r"path/to/your/dataset/metadata.jsonl"

# Get all image files
image_extensions = ['.png', '.jpg', '.jpeg']
image_files = sorted([f for f in os.listdir(images_dir) 
                     if any(f.lower().endswith(ext) for ext in image_extensions)])

print(f"Found {len(image_files)} images to process...")

try:
    with open(output_path, 'w', encoding='utf-8') as outfile:
        processed_count = 0
        
        for img_file in image_files:
            idx = os.path.splitext(img_file)[0]
            
            # Use relative paths for training
            img_path = f"images/{img_file}"
            cond_path = f"conditioning_images/{img_file}"
            text_file = os.path.join(texts_dir, f"{idx}.txt")
            
            # Check files exist
            full_cond_path = os.path.join(conditioning_dir, img_file)
            if not os.path.exists(full_cond_path):
                print(f"⚠️  Missing conditioning image: {img_file}")
                continue
            
            if not os.path.exists(text_file):
                print(f"⚠️  Missing text file: {idx}.txt")
                continue
            
            # Read text description
            try:
                with open(text_file, 'r', encoding='utf-8') as tf:
                    text = tf.read().strip()
                
                if not text:
                    print(f"⚠️  Empty text file: {idx}.txt")
                    continue
                    
            except Exception as e:
                print(f"❌ Error reading {idx}.txt: {e}")
                continue
            
            # Create entry
            entry = {
                "text": text,
                "image": img_path,
                "conditioning_image": cond_path
            }
            
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
            processed_count += 1
            
            if processed_count % 1000 == 0:
                print(f"✅ Processed {processed_count} entries...")
    
    print(f"🎉 Success! Created {processed_count} entries")
    print(f"📁 Output: {output_path}")
    
except Exception as e:
    print(f"❌ Error: {e}")
```

**Run the script:**
```bash
python generate_metadata.py
```

## ⚙️ Training Configuration

### Memory Requirements by GPU

| GPU | VRAM | Batch Size | Gradient Accumulation | Additional Flags |
|-----|------|------------|-----------------------|------------------|
| RTX 3060 Ti | 8GB | 1 | 8 | All memory optimizations |
| RTX 3070/3080 | 8-10GB | 1 | 4-8 | `--gradient_checkpointing` |
| RTX 3080 Ti/3090 | 12GB | 2 | 4 | `--mixed_precision="fp16"` |
| RTX 4080/4090 | 16-24GB | 4 | 2 | `--mixed_precision="bf16"` |

### Key Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--learning_rate` | Learning rate | `1e-5` to `5e-6` |
| `--resolution` | Training resolution | `512` |
| `--num_train_epochs` | Training epochs | `3-10` |
| `--checkpointing_steps` | Save frequency | `5000` |
| `--mixed_precision` | Memory optimization | `"fp16"` or `"bf16"` |

## 🏃‍♂️ Running Training

### For Windows (Without bitsandbytes)

```bash
accelerate launch train_controlnet.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --output_dir="control-model/" \
  --train_data_dir="path/to/your/dataset" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --learning_rate=1e-5 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --set_grads_to_none \
  --num_train_epochs=4 \
  --checkpointing_steps=5000 \
  --enable_xformers_memory_efficient_attention \
  --validation_steps=2500 \
  --report_to="wandb"
```

### For Windows (With Working bitsandbytes)

```bash
# Add this flag if bitsandbytes is working:
--use_8bit_adam
```

## 🐛 Troubleshooting

### Common Windows Issues

| Issue | Solution |
|-------|----------|
| **bitsandbytes error** | Remove `--use_8bit_adam` or install `bitsandbytes-windows` |
| **PowerShell execution policy** | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| **CUDA OOM** | Reduce `--train_batch_size=1`, increase `--gradient_accumulation_steps` |
| **aiohttp error** | `pip install "aiohttp<3.9.0"` |
| **Slow training** | Add `--enable_xformers_memory_efficient_attention` |

### Memory Optimization Flags

```bash
# For low VRAM (8GB)
--train_batch_size=1 \
--gradient_accumulation_steps=8 \
--gradient_checkpointing \
--enable_xformers_memory_efficient_attention \
--mixed_precision="fp16"

# For medium VRAM (12GB)
--train_batch_size=2 \
--gradient_accumulation_steps=4 \
--mixed_precision="fp16"

# For high VRAM (24GB)
--train_batch_size=4 \
--gradient_accumulation_steps=2 \
--mixed_precision="bf16"
```

## 💻 Hardware Recommendations

### Minimum Requirements
- **GPU**: RTX 3060 Ti (8GB VRAM) or better
- **RAM**: 16GB system RAM
- **Storage**: 50GB+ free space for checkpoints
- **OS**: Windows 10/11

### Recommended Setup
- **GPU**: RTX 4080/4090 (16GB+ VRAM)
- **RAM**: 32GB system RAM
- **Storage**: NVMe SSD with 100GB+ free space
- **OS**: Windows 11

## 🎯 Post-Training
Using Your Trained Model in ComfyUI
To use your trained ControlNet model in ComfyUI, follow these steps:

Locate the Model Files:

After training, your model is saved in the control-model/ directory (as specified by --output_dir).
Ensure you have the model checkpoint files (e.g., pytorch_model.bin, config.json) in this directory.


Move the Model to ComfyUI:

Copy the entire control-model/ directory to the ComfyUI models directory, typically located at:ComfyUI/models/controlnet/


Alternatively, place only the necessary model files (pytorch_model.bin, config.json, etc.) into the controlnet folder.


### Load the Model in ComfyUI:

Open ComfyUI and navigate to the node-based workflow interface.
Add a Load ControlNet Model node to your workflow.
In the node settings, select your model from the controlnet directory (it should appear in the dropdown if placed correctly).
Connect the ControlNet node to a Stable Diffusion node, ensuring you provide a control image (e.g., Canny edge map, depth map) that matches the type used during training.

### Test the Workflow:

Run the workflow in ComfyUI to generate an image.
The output should be saved automatically (e.g., as generated.jpg) based on your workflow settings.



### Example ComfyUI Workflow JSON
Below is a sample JSON workflow for using your trained ControlNet model in ComfyUI:


### Steps to Use the Workflow:

Save the above JSON as controlnet_workflow.json.
In ComfyUI, click Load and select the controlnet_workflow.json file.
Ensure the control-model/ directory is in ComfyUI/models/controlnet/.
Update the image_path in the JSON to point to your control image.
Run the workflow to generate the output image.

## 📊 Performance Tips

1. **Dataset Quality** > Quantity - High-quality, diverse data is crucial
2. **Learning Rate**: Start with `1e-5`, adjust based on loss convergence
3. **Validation**: Use `--validation_steps` to monitor training progress
4. **Batch Size**: Larger effective batch size generally improves quality
5. **Training Duration**: 20K-50K steps usually sufficient for good results

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Areas for Contribution
- Additional troubleshooting solutions
- Hardware-specific optimizations
- Dataset preparation tools
- Training script improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) for the training scripts
- [ControlNet](https://github.com/lllyasviel/ControlNet) for the original research
- Community contributors for Windows compatibility solutions

## 📞 Support

If you encounter issues:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Search existing [GitHub Issues](../../issues)
3. Create a new issue with:
   - **System specs** (GPU, RAM, OS)
   - **Error messages** (full traceback)
   - **Training command** used
   - **Dataset info** (size, structure)

## 🎉 Success Stories

Share your trained models and results! Tag them with `#controlnet-windows-training` so others can see what's possible.

---

**Star ⭐ this repo if it helped you train ControlNet on Windows!**
