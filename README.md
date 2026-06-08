# End-to-End Machine Learning Hardware Acceleration on the AMD AUP-ZU3 SoC/FPGA

## Starting!

Documentation is available in the wiki folder

## Overview of the Three Accelerator Projects

This repository contains three independent FPGA accelerator implementations, each designed to illustrate a different class of machine learning models, preprocessing assumptions, and hardware design strategies.
Although they share a common workflow (training → quantization → hls4ml → FPGA wrapper), each project targets a different application domain and showcases a distinct set of techniques.


Together, these three projects demonstrate a broad range of FPGA-accelerable ML systems: from raw signal classification, to image processing, to sequential recommendation systems.


### Gamma/Neutron (G/N) Classifier 

This project is inspired by the work of Molina et al., where the goal is to distinguish between gamma and neutron pulses.
The objective here is to train a compact student model via Knowledge Distillation (KD) so that it can closely match a larger teacher network.

Key features:

- Raw detector-like data is used as input (no images; 1D pulses).

- Signal samples are fed directly to the network, emulating what a real detector would produce.

- The neural architecture is intentionally small, but uses high-precision fixed-point arithmetic (e.g., 16–24 bits) to preserve accuracy.

- Demonstrates that a lightweight FPGA model can achieve performance comparable to the full Keras/QKeras version, thanks to good quantization and KD.

- Objective: 
To show how simple, high-precision, low-latency FPGA models can effectively process raw physical signals.


### MNIST Classifier

This project serves as the canonical 2D image case study using grayscale 28×28 MNIST digits.

Key features:

- Includes the full end-to-end workflow, from dataset normalization → training → quantization → HLS → FPGA.

- The pipeline mirrors the G/N project but uses a 2D convolutional model.

- Demonstrates how to convert image inputs into AXI-Stream format, including fixed-point encoding.

- Useful as a reference for image-based inference running on FPGA hardware.

- Objective: To provide a simple, visual example of the full FPGA deployment process with image data and 2D convolution layers.


### Recommender System

This project focuses on sequential processing and recommendation systems using a synthetic dataset designed to mimic user–item interaction sequences

Key features:

- The workflow starts with a GRU-based model trained for sequential recommendation tasks. The model is then optimized through unstructured pruning, removing low-magnitude weights and reducing parameter count without significantly impacting accuracy.

- After pruning, the network is quantized to 16-bit fixed-point during the hls4ml conversion stage, enabling efficient hardware implementation while retaining numerical stability.

- Post-implementation reports confirm feasible resource usage and accurate inference, evaluated through Top-K and MRR metrics.

- This approach enables efficient FPGA deployment of a sequential recommender.

Objective: To illustrate how sequence models and recommendation systems can be adapted for FPGA inference. 

## How the HLS Acceleration Components Work

These projects contain three key pieces that work together to create an FPGA-ready hardware accelerator:

- *build_accel.tcl*: the automation script that builds the HLS project

- *myproject_embedding_accel.cpp*: the AXI-Stream + AXI-Lite hardware wrapper

- *myproject_embedding_accel.h*: the header file defining the wrapper interface

Together, they integrate the original hls4ml neural network into a fully synthesizable FPGA accelerator block.


| File                                | Purpose           | Role                                                                         |
| ----------------------------------- | ----------------- | ---------------------------------------------------------------------------- |
| **`build_accel.tcl`**               | Automation script | Builds the entire HLS project and synthesizes the accelerator                |
| **`myproject_embedding_accel.cpp`** | AXI wrapper       | Converts AXI stream data ↔ fixed-point tensors, and calls the neural network |
| **`myproject_embedding_accel.h`**   | Interface header  | Defines AXI data type and top-level function signature for HLS               |

Together, they turn a pure C++ neural network from hls4ml into a ready-to-use FPGA accelerator with:

- AXI-Stream input/output

- AXI-Lite control

- DMA compatibility

- Proper fixed-point conversion

- A clean top function for integration in Vivado/Vitis 

<!-- 
hlsPrj_embedding/
    build_accel.tcl	
    firmware/
        myproject.cpp
        myproject.h
        myproject_embedding_accel.cpp
        myproject_embedding_accel.h
        defines.h
        parameters.h
        nnet_utils/
        weights/



PYNQ integration -->


## Software Model Performance Evaluation


| Project                   | Data Type                         | Model Type                  | Hardware Metric Used                                       | Why This Metric?                                                                                              |
| ------------------------- | --------------------------------- | --------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **Gamma / Neutron (G/N)** | 1D raw detector pulses            | Small MLP (KD-trained)      | Confusion Matrix (TP, TN, FP, FN) + per-class accuracy | Binary classification; confusion matrix reveals misclassification structure and class imbalance behavior.     |
| **MNIST**                 | 2D images (28×28)       | CNN                         | Confusion Matrix + overall accuracy + ROC Curve                   | Multi-class classification; confusion matrix shows digit-wise performance and typical confusions (e.g., 4↔9). |
| **RNN / MovieLens**       | Sequential user-item interactions | Student MLP with embeddings | Top-K Accuracy + MRR (Mean Reciprocal Rank)        | Standard metrics for recommender systems; evaluate ranking quality instead of hard labels.                    |


## Hardware Model Performance Evaluation

The metrics differ because each problem requires a different way of measuring model quality, and the hardware evaluation should reflect the true objective of the model.


| Project       | Task Type                  | Best Metric          | Reason                                                   |
| ------------- | -------------------------- | -------------------- | -------------------------------------------------------- |
| G/N           | Binary classification      | Confusion matrix | Captures per-class errors and false positives/negatives  |
| MNIST         | Multi-class classification | Confusion matrix | Reveals digit-specific confusions; easy HW/SW comparison |
| MovieLens RNN | Ranking / recommendation   | Top-K, MRR       | Measures ranking quality rather than hard labels         |





This work was supported in part by the [AMD University Program](https://www.amd.com/en/corporate/university-program.html) 
