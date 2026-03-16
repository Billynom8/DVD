# 🛠️ Installation Guide for DVD for Windows (UV Edition)

This guide provides instructions for installing **DVD: Deterministic Video Depth**. We recommend using the **1-Click Installer** for the easiest experience.

---

## ⚡ Option 1: 1-Click Installer (Recommended)

The easiest way to set up DVD is to use our automated batch script. It handles dependency checks, environment setup, and weight downloads.

### How to get it:
- **If you already downloaded the repo:** Navigate to the `_install` folder and run `DVD_1click_Installer.bat`.
- **If you only want the installer:** Download [DVD_1click_Installer.bat](https://github.com/Billynom8/DVD/raw/main/_install/DVD_1click_Installer.bat) directly, place it in a new folder, and run it. It will clone the rest of the repository for you automatically.

### Running the installer:
1.  Double-click `DVD_1click_Installer.bat`.
2.  Follow the on-screen prompts (it will ask if you want to download model weights).

---

## 🚀 Option 2: Manual Installation (Alternative)

If you prefer to set up your environment manually, follow the steps below.

### 📋 Prerequisites

Ensure the following tools are installed and available in your system's PATH:

- [Git](https://git-scm.com/)
- [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-toolkit)
- [FFMPEG](https://techtactician.com/how-to-install-ffmpeg-and-add-it-to-path-on-windows/)

---

## 🚀 Installation Steps

### 1. Verify CUDA Toolkit Installation

Check that `nvcc` is available and the version is 12.x:

```bash
nvcc --version
```

### 2. Install uv Package Manager

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Clone the Repository & Change Directory Path

```bash
git clone https://github.com/Billynom8/DVD.git
cd DVD
```

### 4. Setup and Install

```bash
uv sync
```

---

## 📦 Model Weights Installation

The models are required to run DVD. You can download them using the Hugging Face CLI.

### 1. Authenticate (Optional but Recommended)
If you haven't logged in to Hugging Face on this machine:

```bash
uv run hf login
```

### 2. Download the Models
Run this command from the `DVD` root folder. The weights will be downloaded to the `ckpt` folder as expected by the inference scripts.

```bash
# Download DVD v1.0 weights
uv run hf download FayeHongfeiZhang/DVD --revision main --local-dir ckpt
```

**Note:** Ensure the final structure matches:
```text
DVD
├── ckpt/
├──── model_config.yaml
├──── model.safetensors
├── ...
```

## ⚡ Optional: SageAttention
For faster inference, you can install SageAttention:

```bash
uv pip install sageattention
```

## ✅ Final Notes

- **Streamlit GUI:** You can launch the interactive web interface simply by running:
  ```bash
  run_app.bat
  ```
- **Inference Scripts:** Command-line batch files for Windows are located in `infer_bash/`.
- **Verify Installation:** Try running `infer_bash\openworld.bat` or launching the GUI to ensure everything is working.
- **Troubleshooting:** If any step fails, check `_install/install_log.txt` (if using the 1-click installer) or verify your CUDA environment.
