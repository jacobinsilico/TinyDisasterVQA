# TinyDisasterVQA on GAP9

Edge visual question answering (VQA) for UAV flood-disaster imagery deployed on the GreenWaves GAP9 microcontroller.

---

## 1. Project Summary

**TinyDisasterVQA** is an efficient edge Visual Question Answering (VQA) system designed for deploying lightweight ML models on the resource-constrained GreenWaves GAP9 microcontroller. The application targets the **FloodNet VQA** dataset to provide automated post-disaster scene analysis (e.g., identifying flooded structures, roads, and counting objects) directly on-device. By replacing heavy natural language processing blocks with mapped template IDs and utilizing highly compressed CNN architectures, the main deployed model (**TDM-XXS single @224**) achieves millisecond-level latency and runs completely locally without requiring cloud connectivity.

---

## 2. Motivation

- **Disaster-Response at the Edge:** Rapid response during severe flooding requires real-time automated situational awareness. UAVs equipped with cameras must process images locally because network infrastructure in disaster zones is typically damaged, non-existent, or highly congested.
- **Resource Constraints of Microcontrollers:** Standard VQA systems combine large vision models (e.g., ResNet-50, Swin Transformers) with sequential language models (e.g., LSTMs, GRUs, or Transformers). These require hundreds of megabytes of RAM and Flash, which far exceeds the limits of ultra-low-power microcontrollers (typically <2 MB SRAM and <16 MB Flash).
- **Simplification Strategy:** TinyDisasterVQA exploits the structured nature of questions in the FloodNet dataset. Instead of executing a heavy recurrent text encoder on the MCU, questions are mapped to a fixed set of templates (`question_template_id`). This permits a simple embedding lookup on-device, shrinking the textual feature extraction stage to negligible memory and computation overhead.

---

## 3. Task Definition

- **Inputs:** 
  1. An aerial image of the disaster scene resized to **224x224 RGB** (quantized on-device to `uint8`).
  2. A question represented by a single integer **`question_template_id`** in the range `[0, 30]`.
- **Output:** A predicted answer from a unified **19-class global vocabulary** (e.g., `"yes"`, `"no"`, `"flooded"`, or numeric counts `"1"`, `"2"`, `"3"`, etc.).
- **VQA vs. Standard Classification:** Normal image classification maps an input image to a single static label. VQA is dynamic; the model must condition its classification on the natural language query. Depending on the question, the same image can output `"yes"` (e.g., "Is the road flooded?") or `"3"` (e.g., "How many buildings are flooded?"). The model dynamically fuses the structural question template embedding with the CNN image feature to produce the correct task prediction.

---

## 4. Model Overview

The student models belong to the **TinyDisasterModel (TDM)** family:

- **Image Encoder:** A custom, ultra-lightweight depthwise-separable CNN. It consists of an entry 2D convolution with Batch Normalization and ReLU (`ConvBNAct`), followed by 4 blocks of `DepthwiseSeparableConv` with stride 2 for spatial downsampling (reducing the resolution from $224 \to 112 \to 56 \to 28 \to 14 \to 7$), and a final `AdaptiveAvgPool2d` layer.
- **Question Encoder:** A template ID lookup embedding (`TemplateQuestionEncoder`) that maps `question_template_id` directly to a small feature vector, bypassing the need for sequential RNN processing.
- **Fusion MLP & Classifier:** Concatenates the image and question features and feeds them into a Multi-Layer Perceptron (MLP) to output logits for the 19 answer classes.
- **Single-Head vs. Multi-Head:**
  - *Single-Head:* Projects the fused features directly into a single 19-class output space. This creates a clean, static execution graph that maps reliably to NNTool.
  - *Multi-Head:* Shares the CNN/embedding fusion trunk but divides the output into 4 distinct heads representing the task domains (`binary`, `condition`, `count`, `density`). During inference, the output is gathered using the corresponding target task ID.

### Key Model Configs & Accuracies
- **Main Deployed Model (`TDM-XXS single @224`):** Features only **13,859 parameters** (~13.5 KB weights) and achieves **80.69% offline test accuracy**. Chosen for GAP9 deployment due to its compact size and straightforward static graph.
- **Best Offline Student (`TDM-XS multihead @224`):** Features **54,516 parameters** (~53.2 KB weights) and achieves **82.60% offline test accuracy**.

---

## 5. Repository Structure

```
TinyDisasterVQA/
├── dataset/                     # Raw annotations and image splits from FloodNet VQA
│   ├── data/                    # JSON files for annotations (train/valid/test), classes, and vocab
│   └── images/                  # Train, validation, and test image folders (.JPG)
├── docs/                        # Offline ablation details
│   ├── teacher_ablation_summary.txt
│   └── student_ablation_summary.txt
├── models/                      # Saved PyTorch checkpoints (.pt) and exported ONNX models (.onnx)
├── outputs/                     # Processed CSV manifests, vocabulary configurations, and summaries
│   ├── answer_space/            # Output classes and loss weights
│   ├── exploration/             # Exploratory dataset distribution statistics
│   ├── processed/               # Shared mappings and manifests
│   └── training_data/           # train.csv, valid.csv, test.csv, and metadata.json
├── runs/                        # PyTorch training run logs and checkpoints
├── scripts/                     # Python scripts for data processing, training, and GAP9 toolchains
│   ├── 01_explore_dataset.py           # Analyzes FloodNet dataset structures
│   ├── 02_build_manifest.py            # Builds unified CSV manifest
│   ├── 03_build_answer_space.py        # Generates class labels and weights
│   ├── 04_prepare_training_data.py     # Tokenizes questions and generates training CSVs
│   ├── 05_train_teacher.py             # Trains teacher model (e.g. ConvNeXt-Tiny)
│   ├── 06_train_student.py             # Distills/trains TDM student variants
│   ├── 07_export_student.py            # Exports PyTorch weights to static ONNX
│   ├── 08_generate_gap9_artifacts.py   # Imports ONNX to NNTool and generates GAP9/Autotiler code
│   └── 09_make_gap9_demo_input.py      # Generates quantized demo inputs (Input_1/2.bin)
├── src/
│   └── tinydisastervqa/         # Main package source code
│       ├── data.py                     # Dataset class and dataloader builder
│       ├── metrics.py                  # VQA and task-wise accuracy calculators
│       ├── models.py                   # PyTorch model architecture definitions
│       └── utils.py                    # Training utilities and helpers
├── gap9_generated/              # Generated C applications, CMake configurations, and hardware parameters
│   └── tdm_xxs_single_ce_224_best/     # Main GAP9 target directory
│       ├── at_model/                   # Generated Autotiler graph source code
│       ├── tensors/                    # Extracted quantized weights for L3 flash load
│       ├── tdm_xxs_single_ce_224_best.c # Main application entry point for GAP9 MCU
│       ├── tdm_xxs_single_ce_224_best.h # Model prefix headers
│       ├── CMakeLists.txt              # GAP SDK build descriptor
│       ├── sdk.config                  # Flash/RAM sizes, clocks, and target settings
│       ├── Input_1.bin                 # Sample image input (uint8 quantized)
│       └── Input_2.bin                 # Sample question input (template ID binary)
└── benchmark_logs_uint8_fixed/  # Real-device logs for latency, cycles, and prediction verification
    └── board_uint8_fixed_run_*.log     # Benchmark runs 1 to 5 on the GAP9 board
```

---

## 6. Dataset and Preprocessing

- **Dataset Split Sizes:**
  - **Train:** 5,898 samples
  - **Validation:** 1,806 samples
  - **Test:** 1,833 samples
  - **Total:** 9,537 samples
- **Image Preprocessing:**
  - *Offline (PyTorch):* Images are resized to `224x224` and normalized using ImageNet statistics: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.
  - *On-Device (GAP9 Input):* To avoid storing costly float32 tensors, the preprocessed image is quantized into 8-bit integers using the NNTool calibration parameters:
    $$\text{scale} = 0.018658448, \quad \text{zero\_point} = 114$$
    The resulting tensor is saved as a 150,528-byte `uint8` file (`Input_1.bin`), achieving a **4.0x storage reduction** over float32.
- **Question Preprocessing:**
  - Text questions are tokenized and matched against 31 predefined structures (templates).
  - The mapped template ID is stored as `Input_2.bin` (8-byte binary file containing the `int64` representation of the ID, which is read on-device).
- **Core Columns Explained:**
  - `target_edge_global`: The ground truth label index in the global 19-class vocabulary.
  - `answer_norm`: The normalized answer text (e.g. `"yes"`, `"flooded"`, `"5"`).
  - `edge_head`: The task category, matching the four heads: `binary`, `condition`, `count`, or `density`.
  - `question_template_id`: The ID mapping of the parsed question text.

---

## 7. Training and Ablation

- **Teacher Training (`scripts/05_train_teacher.py`):** Trains a heavy backbone (e.g. ConvNeXt-Tiny, Swin-Tiny) combined with an LSTM question encoder.
- **Student Training (`scripts/06_train_student.py`):** Trains the tiny TDM variants using standard Cross Entropy (CE) or Knowledge Distillation (KD) from a saved teacher. CE-only training yielded better results, as the teacher's uncertainty on count predictions tended to transfer noise to the students.

### Teacher Ablation Summary
*All teachers fine-tuned end-to-end on `edge_global` using unweighted Cross Entropy.*

| Model Backbone | Resolution | Best Valid Acc | Test Acc | Test Count Acc | Test Density Acc |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ConvNeXt-Tiny (Best Teacher)** | **224** | **89.53%** | **88.11%** | **57.96%** | **84.15%** |
| ConvNeXt-Tiny | 384 | 90.09% | 86.69% | 55.35% | 81.42% |
| Swin-Tiny | 224 | 89.42% | 86.52% | 53.26% | 81.97% |
| Swin-Tiny | 384 | 87.87% | 86.36% | 54.05% | 80.33% |

### Student Ablation Summary
*All students trained from scratch with CE-only at input size 224.*

| Model Variant | Parameters | Int8 Weights | Best Valid Acc | Test Acc | Test Count Acc | Test Density Acc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TDM-XXS single (Deployed)** | **13,859** | **13.5 KB** | **81.89%** | **80.69%** | **36.29%** | **61.20%** |
| TDM-XXS multihead | 17,564 | 17.2 KB | 82.34% | 80.47% | 38.12% | 57.92% |
| TDM-XS single | 47,163 | 46.1 KB | 84.05% | 81.29% | 37.08% | 65.57% |
| **TDM-XS multihead (Best Student)** | **54,516** | **53.2 KB** | **85.66%** | **82.60%** | **39.69%** | **72.68%** |
| TDM-S single | 85,059 | 83.1 KB | 82.72% | 81.45% | 38.90% | 63.93% |
| TDM-S multihead | 96,060 | 93.8 KB | 84.77% | 81.01% | 37.08% | 63.93% |
| TDM-M single | 293,587 | 286.7 KB | 84.33% | 81.89% | 39.69% | 71.04% |

---

## 8. Deployment Pipeline

The full deployment pipeline follows this path:
```
PyTorch Checkpoint (.pt) ──> ONNX ──> NNTool Graph ──> SQ8 Quantization ──> Autotiler ──> CMake GAP9 App ──> GVSoC / Real GAP9 Board
```

- **Pipeline Tool Roles:**
  - `07_export_student.py`: Loads the model checkpoint and exports the PyTorch graph to static `.onnx` with fixed dimensions.
  - `08_generate_gap9_artifacts.py`: Loads the `.onnx` graph into NNTool, runs structural fusions, collects tensor statistics using 32 calibration validation images, applies 8-bit integer quantization (SQ8) using NE16 vector acceleration, and generates Autotiler code.
  - `09_make_gap9_demo_input.py`: Quantizes and converts a test sample into binary inputs (`Input_1.bin` and `Input_2.bin`).
- **Key Generated Artifacts:**
  - `at_model/`: Auto-generated Autotiler graph C files (`model.c`, `Expression_Kernels.c`).
  - `tensors/`: Auto-generated weight files loaded to L3 Flash memory.
  - `Input_1.bin` & `Input_2.bin`: Raw binary tensors representing the image (150.5 KB) and the question ID (8 bytes).
  - `tdm_xxs_single_ce_224_best.c`: Main C driver. Initializes the GAP9 cluster, adjusts clock frequencies and voltages, loads inputs from host filesystem, triggers the Autotiler execution task, and prints cycle stats and predicted classes.
  - `CMakeLists.txt` & `sdk.config`: Standard configuration profiles setting memory allocations and compiler optimizations.

---

## 9. How to Reproduce

*Note: All container-based commands assume the project root is mounted at `/app/TinyDisasterVQA` inside the GAP9 Docker environment.*

### 1. Export Model to ONNX
Run from the repository root in your PyTorch environment:
```bash
PYTHONPATH=src python scripts/07_export_student.py \
  --checkpoint models/tdm_xxs_single_ce_224_best.pt \
  --verify
```

### 2. Launch GAP9 SDK Docker Container
Run the container from your host workstation shell (with WSL/privileged USB passthrough if debugging on-board):
```bash
docker rm -f deeploy_gap9
docker run -it --privileged --name deeploy_gap9 \
  -v ~/Deeploy:/app/Deeploy \
  -v ~/gap9tutorial:/app/gap9tutorial \
  -v ~/TinyDisasterVQA:/app/TinyDisasterVQA \
  -v /dev/bus/usb:/dev/bus/usb \
  ghcr.io/pulp-platform/deeploy-gap9:latest
```

### 3. Initialize GAP9 SDK Environment
Execute inside the running container:
```bash
source /app/install/gap9-sdk/.gap9-venv/bin/activate
source /app/install/gap9-sdk/configs/gap9_evk_audio.sh
export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
```

### 4. Regenerate NNTool and Autotiler Artifacts
Compile the model and generate GAP9 C source files (inside the container):
```bash
cd /app/TinyDisasterVQA
PYTHONPATH=src python scripts/08_generate_gap9_artifacts.py --force
```

### 5. Generate Quantized Demo Inputs
Generate raw binary inputs using validation test index `0`:
```bash
PYTHONPATH=src python scripts/09_make_gap9_demo_input.py --row-index 0
```

### 6. Run on GVSoC (Simulator)
Compile and launch the application on the virtual simulator (inside the container):
```bash
cd /app/TinyDisasterVQA/gap9_generated/tdm_xxs_single_ce_224_best
rm -rf build
cmake -B build -G "Unix Makefiles"
cmake --build build --target menuconfig
# In the menuconfig menu, navigate to: GAP_SDK -> Platform -> Platform -> Select GVSoC
cmake --build build --target run -j$(nproc)
```

### 7. Run on Real GAP9 Board
1. **Host USB Connection:** Bind and attach the physical board to WSL from Windows PowerShell (Run as Administrator):
   ```powershell
   usbipd list
   usbipd bind --busid <BUSID>
   usbipd attach --wsl --busid <BUSID>
   ```
2. **Launch on Hardware:** Build and run the binary on the connected EVK board (inside the Docker container):
   ```bash
   cd /app/TinyDisasterVQA/gap9_generated/tdm_xxs_single_ce_224_best
   rm -rf build
   cmake -B build -G "Unix Makefiles"
   cmake --build build --target menuconfig
   # In the menuconfig menu, navigate to: GAP_SDK -> Platform -> Platform -> Select Board
   cmake --build build --target run -j$(nproc)
   ```

---

## 10. Benchmark Results

### Execution Statistics (TDM-XXS single @224)
*Aggregated performance results recorded over 5 separate physical board runs.*

| Metric | Target Value |
| :--- | :--- |
| Board Frequency | 370 MHz |
| Cluster Operating Voltage | 800 mV |
| Total Executed MAC Operations | 4,392,832 ops |
| Mean Execution Cycles | 782,037.2 cycles |
| **Mean Inference Latency** | **2.1136 ms** |
| Mean Performance Efficiency | 5.6172 ops/cycle |
| Throughput | ~473 inferences/sec |
| Output Stability | 5 / 5 runs identical (argmax: 1) |

### Memory Configuration & Static Sizes

| Allocation Segment | Config / File Size | Memory Space |
| :--- | :--- | :--- |
| Model Parameter Count | 13,859 | - |
| Int8 Static Weights Size | 13.5 KB | MRAM / Flash |
| Total Working Tensors | 232 KB | SRAM L2 / L1 |
| Binary Executable Size | 266,614 bytes (ELF) | Text: 107.6 KB, Data: 153.2 KB, BSS: 5.8 KB |
| Input Image (`Input_1.bin`) | 150,528 bytes | L2 RAM (uint8) |
| Input Question (`Input_2.bin`) | 8 bytes | L2 RAM (int64) |
| L1 Cluster Memory Limit | 128,000 bytes | `CONFIG_MODEL_L1_MEMORY` |
| L2 SoC Memory Limit | 1,200,000 bytes | `CONFIG_MODEL_L2_MEMORY` |
| L3 External Memory Limit | 8,000,000 bytes | `CONFIG_MODEL_L3_MEMORY` |

---

## 11. Demo Sample Verification

To verify functional execution across frameworks, a single sample was isolated and tracked from PyTorch to the real GAP9 hardware:

- **Image Path:** `dataset/images/test_images/10163.JPG`
- **Question:** `"Is the area mostly non flooded"`
- **Ground Truth Answer:** `"yes"` (class label index `1`)
- **Execution Outputs:**

| Framework / Hardware | Predicted Class | Raw Confidence Score |
| :--- | :---: | :---: |
| **PyTorch (FP32)** | **1** | Logit: 5.1187 |
| **ONNX Runtime (FP32)** | **1** | Logit: 5.1187 |
| **GAP9 Board Execution (Int8)** | **1** | Logit: 176 (uint8 scale) |

*The output logits are printed as `uint8` arrays from the quantized model, and the argmax classification is calculated directly on the GAP9 processor.*

---

## 12. Limitations

- **Offline Accuracy Evaluation:** Test accuracy (80.69%) was measured using PyTorch on the host system. The full test split was not compiled and run end-to-end on the microcontroller itself.
- **Power & Energy Analysis:** Dynamic current draw and active energy consumption (e.g., millijoules per inference) were not measured using hardware probes in the current minimum viable version.
- **Sub-optimal Deployed Model:** The multi-head student model (`TDM-XS multihead`) achieves better accuracy offline (82.60%) than the deployed model, but is not currently supported in the GAP9 deployment directory due to the added complexity of multi-branch ONNX output routing.

---

## 13. Citations and Acknowledgments

- **FloodNet Dataset Reference:**
  ```bibtex
  @inproceedings{rahnemoonfar2021floodnet,
    title={Floodnet: A high resolution aerial imagery dataset for post disaster damage assessment},
    author={Rahnemoonfar, Maryam and Chowdhury, Tateona and Sarkar, Robin and Varshney, Debvrat and Yasar, Masood and Ekblad, David},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={13587--13597},
    year={2021}
  }
  ```
- **TinyVQA Baseline Reference:**
  ```bibtex
  @inproceedings{alrashid2024tinyvqa,
    title={TinyVQA: Compact Multimodal Deep Neural Network for Visual Question Answering on Resource-Constrained Hardware},
    author={Al Rashid, Hasib and Sarkar, Argho and Gangopadhyay, Aryya and Rahnemoonfar, Maryam and Mohsenin, Tinoosh},
    booktitle={Proceedings of the tinyML Research Symposium},
    year={2024}
  }
  ```