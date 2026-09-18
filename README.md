# 📡 Presence Detection with Ultra-Wide Band (UWB) Radar

This repository contains the code, models, and documentation for a TinyML project focused on real-time human presence detection and localization using Ultra-Wide Band (UWB) radar. The pipeline integrates radar signal preprocessing, Digital Beamforming (DBF), and a lightweight deep learning object detection model optimized for resource-constrained edge devices. 

## 🛠️ Hardware & Data Collection
The system was developed and tested using two different UWB radars:
* **TrueSense SR250** (1 TX, 3 RX antennas)
* **Infineon PWMC** (1 TX, 3 RX antennas)

**📊 Dataset Overview:**
* 90 samples were collected across 9 fixed positions in a grid (distances of 1.5m, 2.5m, and 3.5m at angles of -22.5°, 0°, and 22.5°).
* Scenarios include up to 4 people standing still for 1 minute, as well as single targets moving between positions every 30 seconds.

## ⚙️ Signal Preprocessing & Beamforming Pipeline
To translate raw time-domain radar data into a spatial representation suitable for computer vision models, the following pipelines are implemented:

### 1. Decluttering
Static environmental reflections are removed using either a **Simple Moving Average (SMA)** (functioning as an FIR filter) or an **Exponentially-Weighted Moving Average (EWMA)** (functioning as an IIR filter).

### 2. Infineon-Specific Preprocessing
Due to the specific ADC data structure of the Infineon radar, raw data undergoes additional processing before beamforming:
* **Normalization** to a $[-1, 1]$ range.
* **DC Removal** (bias correction per frame/chirp/antenna).
* **Blackman-Harris Windowing** to reduce leakage across range bins.
* **Zero-Padding** to increase Fast Fourier Transform (FFT) resolution.
* **Range FFT** and **Chirp Averaging** to convert the time-domain signal into a frequency/range profile while minimizing noise.

### 3. Digital Beamforming (DBF)
A spatial filter is applied to pinpoint the azimuth angle of the targets. Complex weights are calculated and applied to the signals from multiple receiver antennas to produce a 3D tensor (time, range, angle) mapped across a $[-45^\circ, +45^\circ]$ field of view. 

## 🎯 Object Detection
The continuous 3D radar stream is aggregated via a sliding time window (e.g., 40 frames with a step size of 20) and coherently averaged to generate 2D Range-Azimuth heatmaps: these heatmaps serve as the input dataset.

* **🧠 Model Architecture:** We utilise **FOMO** (Faster Objects, More Objects), a lightweight object detection model built on a MobileNetV2 backbone (0.1 and 0.35 alpha variations). FOMO operates by dividing the image into a grid and predicting centroid locations, eliminating the computational overhead of traditional anchor-box detectors.
* **🖼️ Input Formats:** The models are trained on $96\times96$ and $64\times64$ grayscale and RGB images with axes removed to maintain spatial consistency.
* **📈 Training Strategy:** Models were evaluated on both a 9-class formulation (predicting specific grid positions) and a 1-class formulation ("person" vs. background).

## 🚀 Performance & Deployment
The 1-class object detection approach yields significantly higher precision and recall than the 9-class approach, proving highly effective for generalizing target detection.

The resulting quantized (int8) FOMO models are highly optimized for microcontroller deployment via the Edge Impulse EON Compiler or TensorFlow Lite:
* **MobileNetV2 0.35 (int8):** ~811 ms latency | ~239.4 KB RAM | ~70.7 KB Flash
* **MobileNetV2 0.1 (int8):** ~751 ms latency | ~235.5 KB RAM | ~58.4 KB Flash

## 👥 Contributors
* Valeria de Gennaro
* Matteo Figini
* Tiya Jreige

*Hardware Architectures for Embedded and Edge AI (2024/25) - Politecnico di Milano*
