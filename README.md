<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:4F46E5,100:10B981&height=220&section=header&text=FPGA%20CNN&fontSize=50&fontColor=ffffff&animation=twinkling&fontAlignY=35&desc=Brain%20Tumor%20Detection%20on%20FPGA%20using%20Convolutional%20Neural%20Networks&descSize=20&descAlignY=60&descColor=ffffff" width="100%"/>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=2500&pause=800&color=00D4FF&center=true&vCenter=true&multiline=true&width=800&height=80&lines=⚡+Hardware-Accelerated;📡+Real-Time+CNN+Inference+for+Medical+Diagnostics)](https://git.io/typing-svg)

<br/>

[![Made with FPGA](https://img.shields.io/badge/Made%20with-FPGA-blueviolet?style=for-the-badge&logo=chip&logoColor=white)](#)
[![CNN](https://img.shields.io/badge/Model-CNN-orange?style=for-the-badge&logo=tensorflow&logoColor=white)](#)
[![Verilog](https://img.shields.io/badge/HDL-Verilog-informational?style=for-the-badge&logo=v&logoColor=white)](#)
[![Vivado/Quartus](https://img.shields.io/badge/Toolchain-Vivado%20%7C%20Quartus%20Prime-red?style=for-the-badge&logo=xilinx&logoColor=white)](#)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](#-license)

</div>

---

## 🔬 Abstract

This project delivers a hardware-accelerated solution for brain tumor detection by implementing a **Convolutional Neural Network (CNN) on FPGA hardware**. By exploiting the parallel-processing capabilities of FPGAs, the system supports real-time CNN inference with improved efficiency, while the flexibility of the FPGA fabric allows intricate CNN models to be mapped onto hardware for faster processing and reduced power consumption — key requirements for healthcare scenarios where rapid response times matter. The approach trains a CNN on brain MRI datasets, extracts the learned weights and biases, and implements the network layer-by-layer on FPGA — beginning with a single neuron as an initial validation step and progressing to the full pipeline. The project evaluates and compares implementations across CPU, GPU, and FPGA platforms, highlighting FPGA's superior speed and energy efficiency for portable diagnostics in resource-constrained environments.

<div align="center">
<img src="docs/images/block_diagram.png" alt="Block diagram of the FPGA CNN brain tumor detection pipeline" width="850"/>
</div>

<details>
<summary><strong>🖼️ Click to view the equivalent flow as a Mermaid diagram</strong></summary>

```mermaid
flowchart TD
    A[1. MRI Brain Image Acquisition] --> B[2. Image Preprocessing<br/>Resizing • Normalization • Noise Filtering • Intensity Adjustment]
    B --> C[3. Trained CNN Model]
    C --> D[4. Quantization & Optimization<br/>Fixed-Point Q8.8 Conversion • Weight Quantization • Pruning • Compression]
    D --> E[5. FPGA Implementation<br/>Parallel Processing Pipeline]
    E --> F[6. Tumor Classification<br/>Tumor Detected / No Tumor Detected]
    F --> G[7. Real-Time Display & Output<br/>Alert / Report Generation]
```

</details>

---

## 🎯 Motivation & Problem Statement

Convolutional Neural Networks have proven highly effective for image classification and pattern recognition, making them valuable for medical imaging tasks such as locating tumors in MRI scans. However, classical CNN inference on CPUs and GPUs carries high latency, high power demand, and limited scalability — significant drawbacks for real-time clinical diagnosis.

This project addresses that gap by implementing CNNs on **Field Programmable Gate Arrays (FPGAs)**, which process data in parallel and allow custom logic tailored to the CNN topology, improving both speed and power efficiency. The goal is a real-time, high-performance, energy-efficient brain tumor detection system capable of managing the computational demands of CNN inference within the practical constraints of healthcare diagnostics — designing, training, and deploying a CNN architecture on FPGA hardware to deliver a resource-efficient, scalable solution for precise and rapid MRI-based tumor detection.

### Research Gaps Identified

- **Performance bottlenecks** — limitations in processing speed/efficiency of CNNs on FPGA relative to CPU/GPU, and resource pressure on LUTs, DSPs, and memory.
- **Scalability issues** — challenges in scaling CNN architectures across different network sizes, layer types, and input variations while managing FPGA resource allocation.

## 🌍 Relevance to Sustainable Development Goals

The project aligns with the following UN Sustainable Development Goals:

| SDG | Alignment |
|-----|-----------|
| **Goal 3 — Good Health and Well-being** | Faster, more accurate tumor detection enables early intervention and improved patient care. |
| **Goal 9 — Industry, Innovation and Infrastructure** | Advances FPGA technology for AI applications, promoting responsible innovation and resilient infrastructure. |
| **Goal 11 — Sustainable Cities and Communities** | Energy-efficient, real-time processing extends to broader smart-city and urban-system applications. |

---

## ✅ Objectives

- **Real-Time Processing** — implement a hardware-accelerated CNN on FPGA for real-time analysis of brain MRI images.
- **High Accuracy and Efficiency** — design a CNN architecture with high diagnostic accuracy while balancing power consumption and computational performance on FPGA.
- **Scalable Hardware Design** — support deployment across various FPGA platforms and healthcare workloads.
- **Integration with Diagnostic Tools** — align the hardware implementation with practical diagnostic workflows for seamless integration with existing medical imaging systems.
- **Resource Optimization** — apply quantization, pruning, and efficient parameter storage to maximize FPGA resource utilization.
- **Sustainability and Cost-Efficiency** — prioritize energy efficiency to reduce environmental impact and enable deployment in low-resource settings.

## 📖 Literature Review

The intersection of FPGA hardware acceleration and CNN-based deep learning has drawn substantial research interest, particularly for real-time inference. Recurring themes across the literature include:

- FPGA-based implementations consistently show **energy efficiency and latency advantages** over CPU/GPU inference, largely due to parallel processing of convolutional layers.
- **Quantization** (reducing weight/activation bit-width) and **pruning** (removing redundant parameters) are the dominant optimization strategies for fitting CNNs within FPGA resource constraints (LUTs, BRAM, DSP slices).
- **Hardware-software co-design** and hardware-aware training help tailor CNN models specifically for FPGA deployment.
- **High-level synthesis (HLS) tools**, such as Xilinx Vivado HLS, have streamlined converting neural network models into hardware implementations.
- Open challenges remain around **scalability and generalization** when adapting large-scale CNNs to FPGA platforms; hybrid FPGA + cloud/edge approaches are suggested as a path forward.

---

## 🏗 System Architecture

### CNN Model Architecture (Software / Training)

```mermaid
flowchart LR
    IMG[Input MRI Image] --> C1[Convolution + Activation]
    C1 --> P1[Pooling]
    P1 --> C2[Convolution + Activation]
    C2 --> P2[Pooling]
    P2 --> VEC[Vectorised Feature Maps]
    VEC --> FC[Fully Connected Layer]
    FC --> OUT[Classification Output]
```

### FPGA Hardware Pipeline (`cnn_top` — as implemented in Verilog)

The FPGA implementation processes a **flattened 64×64 grayscale MRI image** (8 bits/pixel, 32,768 bits total) through the following pipeline:

```mermaid
flowchart TD
    IN[Input Image<br/>64x64, 8-bit grayscale] --> CV1[Convolution Layer 1<br/>3x3 kernel → 62x62 feature map]
    CV1 --> R1[ReLU Activation]
    R1 --> MP1[Max Pooling 1<br/>2x2, stride 2 → 31x31]
    MP1 --> CV2[Convolution Layer 2<br/>3x3 kernel on 16-bit fixed point → 29x29]
    CV2 --> R2[ReLU Activation]
    R2 --> MP2[Max Pooling 2<br/>2x2, stride 2 → 14x14]
    MP2 --> GAP[Global Average Pooling<br/>196 elements averaged → 1 scalar]
    GAP --> FC[Fully Connected Layer<br/>x225 weights + bias]
    FC --> CLS[Classification Decision Logic<br/>4-bit encoded class output]
```

**Legend:** Weights and biases for each layer are read from external memory files via `$readmemh` and stored in internal registers for runtime use.

---

## 🧪 Methodology / Design Approach

The design follows a structured progression from software training to FPGA deployment:

| Stage | Description |
|-------|-------------|
| **1. Data Collection & Preprocessing** | Brain MRI scans (labeled tumor/no-tumor) are collected. Images are converted to grayscale, resized to a uniform dimension, and pixel intensities normalized to improve training efficiency. |
| **2. CNN Model Design & Training** | A CNN with multiple convolutional, pooling, and fully connected layers is built and trained in **TensorFlow** (on Google Colab), using backpropagation and gradient descent. Accuracy, precision, recall, and F1-score are tracked during training. |
| **3. Extraction of Weights & Biases** | Post-training, weights and biases are extracted from each layer and converted to **Q8.8 fixed-point** format for hardware compatibility. |
| **4. FPGA System Modeling** | The CNN layers (convolution, pooling, fully connected) are translated into FPGA-compatible Verilog modules. Neuron-level validation confirms hardware computations match software outputs. |
| **5. CNN Hardware Mapping** | The full architecture is mapped onto the FPGA by replicating neurons per layer and connecting layers to mirror the TensorFlow design, with optimizations for parallelism and reduced resource usage. |
| **6. Verification & Validation** | FPGA outputs are compared against the software CNN model to confirm functional correctness and timing alignment. |
| **7. Performance Analysis** | Classification accuracy, latency, power consumption, and FPGA resource usage (LUTs, flip-flops, DSP slices, BRAM) are measured and benchmarked against CPU/GPU implementations. |
| **8. Real-Time Testing & Refinement** | The system is tested with new MRI scans in a practical setting, with refinements applied to improve accuracy, speed, and overall performance. |

<details>
<summary><strong>FPGA Design Flow (synthesis to bitstream)</strong></summary>

1. **High-Level Design** — CNN architecture converted into hardware-friendly components (convolution, pooling, fully connected mapped to FPGA modules).
2. **Behavioral Simulation** — high-level simulation verifies each CNN layer's functionality against the software model.
3. **Synthesis** — Verilog code synthesized (Xilinx Vivado / Intel Quartus) into a gate-level netlist.
4. **Implementation** — design mapped onto physical FPGA resources (LUTs, Flip-Flops, DSPs, Block RAM) under timing constraints.
5. **Place and Route** — components placed and routed on the FPGA fabric to minimize delay and maximize performance.
6. **Configuration File Generation** — a bitstream is generated to program the FPGA.

</details>

---

## 🔢 Q8.8 Fixed-Point Representation

FPGA hardware typically lacks efficient floating-point support, so weights and biases are represented using **Q8.8 fixed-point** format — a 16-bit word split into 8 integer bits and 8 fractional bits.

| Property | Value |
|----------|-------|
| **Total word length** | 16 bits (Q8.8) |
| **Value range** | −128.0 to +127.99609375 |
| **Resolution** | 2⁻⁸ = 0.00390625 |
| **Representation** | Two's complement (signed) |

<details>
<summary><strong>Worked conversion examples</strong></summary>

**Decimal → Q8.8**

```
6.5 × 256 = 1664  →  Q8.8: 0x0680
```

**Q8.8 → Decimal**

```
0x01A0 = 416  →  416 ÷ 256 = 1.625
```

**Representing +5.25**
```
Integer: 5 → 00000101
Fraction: 0.25 × 256 = 64 → 01000000
Combined: 00000101 01000000 → 0x0540
```

**Representing −3.75 (two's complement)**
```
Integer: 3 → 00000011
Fraction: 0.75 × 256 = 192 → 11000000
Positive form: 00000011 11000000
Invert + 1: 11111100 01000000 → 0xFC40
```

</details>

**Why Q8.8:** it requires less hardware than floating-point units, enables faster and more deterministic arithmetic, but trades off dynamic range and precision — an acceptable tradeoff for low-precision neural network inference on resource-constrained FPGAs.

---

## ⚙️ Verilog Module Breakdown

| Module | Responsibility |
|--------|-----------------|
| **`cnn_top`** | Top-level module coordinating the full CNN pipeline: Conv1 → ReLU → Pool1 → Conv2 → ReLU → Pool2 → Global Average Pooling → Fully Connected → Classification Decision Logic. |
| **`conv2d`** | Performs convolution over a 3×3 pixel window and corresponding weights; pixel values extended to 16 bits to match weight precision; multiply-accumulate is sequenced via an FSM across clock cycles; bias added and result scaled. |
| **`relu`** | Takes a 16-bit signed value; outputs zero if negative, otherwise passes the value through — introducing non-linearity. |
| **`max_pooling`** | Compares a 2×2 input window sequentially and stores/outputs the maximum value. |
| **`fully_connected`** | Multiplies the single scalar from global average pooling with 225 weights, sums with a bias (accumulated in a 32-bit register), then scales down to a 16-bit output. |

**Weight & bias storage:** extracted parameters are stored in **Block RAM (BRAM)** on the FPGA for fast access during inference, minimizing latency and improving memory efficiency.

**Resource optimization techniques used:**
- **Quantization** — reduces bit-width of weights/biases to cut memory and compute requirements.
- **Pruning** — removes redundant parameters to reduce model size without sacrificing accuracy.

<div align="center">
<img src="docs/images/rtl_schematic.png" alt="RTL schematic of the synthesized CNN hardware design" width="750"/>
<br/>
<sub><strong>Fig.</strong> RTL schematic of the synthesized CNN design, generated post-synthesis in the FPGA toolchain.</sub>
</div>

---

## 🧰 Hardware & Software Tools

<div align="center">

| Category | Tool | Purpose |
|----------|------|---------|
| **Model Development** | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) via **Google Colab** | CNN design and training with GPU acceleration |
| **Fixed-Point Conversion** | **MATLAB** | Converting extracted weights/biases to Q8.8 fixed-point format |
| **Hardware Description** | **Verilog HDL** | Defining and connecting CNN layers as FPGA modules |
| **FPGA Synthesis** | ![Xilinx Vivado](https://img.shields.io/badge/Xilinx-Vivado-critical?style=flat-square&logo=xilinx&logoColor=white) / **Quartus Prime** | Synthesis, simulation, and programming of the CNN hardware design |
| **Target Board** | **Digilent Genesys 2 FPGA Board** | Selected for its parallel processing capability, DSP slices, and Block RAM |

</div>

---

## 📊 Results & Performance

### Per-Class Classification Accuracy

<div align="center">
<img src="docs/images/accuracy_chart.png" alt="Per-class classification accuracy chart" width="550"/>
</div>

| Tumor Class | Accuracy |
|-------------|----------|
| Glioma | 0.80 |
| Meningioma | 0.90 |
| No Tumor | 0.90 |
| Pituitary | 0.70 |

### Summary Performance Metrics

<div align="center">

| Metric | Value / Outcome | Remarks |
|--------|------------------|---------|
| **Model Accuracy** | **94.2%** | Slight reduction due to quantization and pruning on FPGA |
| **Processing Latency** | **~35 ms** per MRI image | Real-time processing capability achieved |
| **Power Consumption** | **~1.8 W** | Significantly lower than GPU-based implementations |
| **FPGA Resource Utilization** | LUTs: 78% • FFs: 71% • BRAMs: 64% • DSPs: 85% | Efficient use of FPGA resources with room for scaling |
| **Model Size after Optimization** | **1.3 MB** | Reduced via quantization and pruning |
| **Hardware Accuracy Deviation** | **±1.3%** | Relative to the original floating-point software model |

</div>

> [!NOTE]
> These summary metrics are as reported in the project poster. The verification methodology compared FPGA output against the software model via simulation and real-time testing with new MRI scans (see [Methodology](#-methodology--design-approach)).

### Validation Approach

- **Software vs. Hardware** — the FPGA-based model's output was validated against the Google Colab–trained software model, with results closely matching.
- **Compilation & Simulation** — the design compiled without major syntax/semantic errors; simulation with multiple test cases confirmed functional correctness and timing alignment.
- **Real-Time Processing** — the FPGA implementation demonstrated real-time capability with minimal latency.

---

## 📁 Repository Structure

> [!IMPORTANT]
> The structure below is a **suggested organization** for this repository. Update it to match your actual folder/file layout.

```text
fpga-cnn-brain-tumor-detection/
├── docs/
│   ├── images/
│   │   ├── block_diagram.png      # System block diagram (used in this README)
│   │   ├── rtl_schematic.png      # Post-synthesis RTL schematic (used in this README)
│   │   └── accuracy_chart.png     # Per-class accuracy chart (used in this README)
│   ├── Flow_chart.pdf             # Project poster / summary
│   └── PROJECT_REPORT.docx        # Full project report
├── model/
│   ├── cnn_training.ipynb         # CNN design & training (Google Colab / TensorFlow)
│   ├── weight_extraction.m        # Weight/bias extraction & Q8.8 conversion (MATLAB)
│   └── saved_model/               # Trained model artifacts
├── fpga/
│   ├── rtl/
│   │   ├── cnn_top.v               # Top-level pipeline module
│   │   ├── conv2d.v                 # Convolution module
│   │   ├── relu.v                   # ReLU activation module
│   │   ├── max_pooling.v            # Max pooling module
│   │   └── fully_connected.v        # Fully connected layer module
│   ├── mem/                       # $readmemh weight/bias files
│   ├── testbenches/                # Verification testbenches
│   └── constraints/                # Timing/pin constraint files (Genesys 2)
├── presentation/
│   └── ppt.pptx                   # Project presentation slides
├── results/
│   └── accuracy_metrics.md        # Per-class accuracy & benchmark results
├── LICENSE
└── README.md
```

---

## 🚀 Getting Started

> [!NOTE]
> These are template setup steps — adjust commands, paths, and board part numbers to match your actual codebase.

<details>
<summary><strong>Prerequisites</strong></summary>

- Python 3.x with TensorFlow (CNN training)
- MATLAB (weight/bias Q8.8 conversion)
- Xilinx Vivado (WebPACK) or Intel Quartus Prime
- Digilent Genesys 2 FPGA board (or equivalent)
- Labeled brain MRI dataset (tumor / no-tumor, with tumor subtypes)

</details>

<details>
<summary><strong>1. Clone the repository</strong></summary>

```bash
git clone https://github.com/your-username/fpga-cnn-brain-tumor-detection.git
cd fpga-cnn-brain-tumor-detection
```

</details>

<details>
<summary><strong>2. Train the CNN model</strong></summary>

```bash
# Run in Google Colab or locally with TensorFlow
python model/cnn_training.py
```

</details>

<details>
<summary><strong>3. Extract and convert weights/biases to Q8.8</strong></summary>

```matlab
% MATLAB: convert trained weights/biases to Q8.8 fixed-point .mem files
run('model/weight_extraction.m')
```

</details>

<details>
<summary><strong>4. Simulate and synthesize the Verilog design</strong></summary>

```bash
# Open the RTL project in Vivado or Quartus Prime, run behavioral simulation,
# then synthesis, implementation, and bitstream generation
```

</details>

<details>
<summary><strong>5. Program the FPGA and run inference</strong></summary>

```text
Load the bitstream onto the Genesys 2 board, stream a preprocessed
64x64 grayscale MRI image into cnn_top, and read the 4-bit encoded
classification result.
```

</details>

---

## ⚠️ Limitations & Constraints

- **Resource limitations on FPGA** — the Genesys 2 board's restricted resources required quantization and pruning, which could slightly impact overall accuracy.
- **Fixed-point arithmetic precision** — Q8.8 quantization reduces precision relative to floating-point software CNNs, resulting in minor accuracy deviations.
- **Complexity in hardware mapping** — translating the CNN architecture into Verilog is labor-intensive and requires expertise, particularly when validating layer functionality for real-time performance.
- **Dataset limitations** — model performance and generalizability depend on the quality and variability of the MRI dataset used.

## 🔭 Future Work

- **Advanced FPGA platforms** — higher-capacity boards with more DSP slices and memory to reduce reliance on quantization.
- **Refined optimization techniques** — advanced model compression, improved quantization, or sparsity-promoting training.
- **Diverse and extensive datasets** — broader MRI datasets across tumor types and brain regions to improve adaptability.
- **Hybrid cloud integration** — combining FPGA hardware with cloud computing for flexible, scalable hybrid systems.
- **Clinical environment testing** — real-world trials with live MRI data to assess practical utility and diagnostic effectiveness.

---

## 🌱 Social & Environmental Impact

**Social:**
- **Healthcare innovation** — real-time MRI tumor detection has life-saving potential through early, reliable diagnoses, and can extend diagnostic access to remote and underserved areas.
- **Safety in autonomous systems** — the same FPGA-accelerated CNN approach can improve real-time decision-making in domains like autonomous vehicles and robotics.
- **Economic and employment growth** — increasing adoption of AI hardware solutions drives demand for expertise in AI research, FPGA design, and software engineering.

**Environmental:**
- **Energy efficiency** — FPGA implementations are significantly more energy-efficient than CPU/GPU counterparts, supporting sustainable large-scale AI deployment.
- **Resource optimization** — quantization, pruning, and efficient memory management reduce the environmental footprint of high-complexity AI models.

## 💰 Cost Analysis

No direct expenses were incurred in this project:

| Category | Cost Basis |
|----------|-----------|
| **Hardware** | FPGA board and components provided by the institution |
| **Software** | Google Colab and MATLAB used free of charge; FPGA tools used via free/WebPACK editions |
| **Additional Resources** | Research papers and documentation accessed through institutional resources |

The project was completed at **zero cost**, demonstrating the feasibility of FPGA-based CNN projects within a budget-free setup.

## 🏁 Conclusion

This project designed and implemented a CNN on FPGA hardware for brain tumor detection with real-time processing capability. The workflow spanned data acquisition and preprocessing, CNN architecture development and training (TensorFlow / Google Colab), extraction and Q8.8 fixed-point conversion of model parameters (MATLAB), and hardware deployment in Verilog on a Genesys 2 FPGA board (via Xilinx Vivado). By leveraging parallel processing, the FPGA system achieves low-latency operation well-suited to real-time medical diagnostic applications, with evaluations showing strong performance across accuracy, latency, power efficiency, and resource utilization compared to traditional CPU/GPU implementations.

---

## 📚 References

1. Md. Najrul Islam, Rahul Shrestha, Shubhajit Roy Chowdhury. *"Energy-Efficient and High-Throughput CNN Inference Engine Based on Memory-Sharing and Data-Reusing for Edge Applications."* IEEE Transactions on Circuits and Systems.
2. Tao Jiang, Ligang Xing, Jinming Yu, Junchao Qian. *"A hardware-friendly logarithmic quantization method for CNNs and FPGA implementation."* IEEE Transactions on Circuits and Systems.
3. Sertaç Yaman, Barış Karakaya, Yavuz Erol. *"A novel normalization algorithm to facilitate pre-assessment of Covid-19 disease by improving accuracy of CNN and its FPGA implementation."* IEEE Transactions on Biomedical Engineering.
4. Sai Sanjeet, Bibhu Datta Sahoo, Masahiro Fujita. *"Energy-Efficient FPGA Implementation of Power-of-2 Weights-Based Convolutional Neural Networks With Low Bit-Precision Input Images."* IEEE Transactions on Circuits and Systems.
5. Rizwan Tariq Syed, Yanhua Zhao, Junchao Chen, Marko Andjelkovic, Markus Ulbricht, Milos Krstic. *"FPGA Implementation of a Fault-Tolerant Fused and Branched CNN Accelerator With Reconfigurable Capabilities."* IEEE Transactions on Circuits and Systems.
6. Ibrahim Fennouh, Said Sadoudi, Djamal Teguig, Camel Tanougast. *"FPGA implementation of orthogonal hyperchaotic sequences generator based on the CNN: application in multi-user chaotic communications."* IEEE Transactions on Circuits and Systems.
7. Daniel Enériz, Antonio J. Rodriguez-Almeida, Himar Fabelo, Samuel Ortega, Francisco J. Balea-Fernandez, Gustavo M. Callico, Nicolás Medrano, Belén Calvo. *"Low-Cost FPGA Implementation of Deep Learning-Based Heart Sound Segmentation for Real-Time CVDs Screening."*
8. Hiroshi Fuketa, Toshihiro Katashita, Yohei Hori, Masakazu Hioki. *"Multiplication-Free Lookup-Based CNN Accelerator Using Residual Vector Quantization and Its FPGA Implementation."*
9. Gianmarco Baldini, Fausto Bonavitacola, Jean-Marc Chareau. *"Wireless Interference Identification With Convolutional Neural Networks Based on the FPGA Implementation of the LTE Cell-Specific Reference Signal (CRS)."*
10. Chen Yang, Yizhou Wang, Xiaoli Wang, Li Geng. *"Energy-Efficient and High-Throughput CNN Inference Engine Based on Memory-Sharing and Data-Reusing for Edge Applications."*
11. Darío Baptista, F. Morgado-Dias, Leonel Sousa. *"A Platform based on HLS to Implement a Generic CNN on an FPGA."* 2019 International Conference in Engineering Applications (ICEA).
12. Meriam Dhouibi, Ahmed Karim Ben Salem, Slim Ben Saoud. *"CNN for object recognition implementation on FPGA using PYNQ framework."* 2020 IEEE Eighth International Conference on Communications and Networking (ComNet).
13. Federico G. Zacchigna. *"Methodology for CNN Implementation in FPGA-Based Embedded Systems."* IEEE Embedded Systems Letters.
14. Muhammad Arbab Arshad, Sakib Shahriar, Assim Sagahyroon. *"On the Use of FPGAs to Implement CNNs: A Brief Review."* 2020 International Conference on Computing, Electronics & Communications Engineering (iCCECE).

## 👤 Author

**Haridharan K S** 
B.Tech, Electronics and Communication Engineering
Vellore Institute of Technology (VIT), Chennai
Supported by **V-NEST** (VIT Chennai Startup and Research Foundation)

## 📄 License

This project is released under the [MIT License](LICENSE). Update this section if a different license applies.

---

<div align="center">

<sub>⭐ If you find this project useful, consider starring the repository!</sub>

</div>
