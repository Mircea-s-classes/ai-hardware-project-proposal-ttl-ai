# University of Virginia
## Department of Electrical and Computer Engineering

**Course:** ECE 4332 / ECE 6332 — AI Hardware Design and Implementation  
**Semester:** Fall 2025  
**Proposal Deadline:** November 5, 2025 — 11:59 PM  
**Submission:** Upload to Canvas (PDF) and to GitHub (`/docs` folder)

---

# AI Hardware Project Proposal

## 1. Project Title
**Edge-Deployed Vision Bubble Detection for MTB Brake Lines**  
**Team:** TTL AI

**Students:**  
Landon Campbell  
Thomas Keyes  
Tiger Zhang

---

## 2. Platform Selection
**Category (Undergraduates):** **Edge-AI** (camera-based)  
**Chosen Platform:** **Raspberry Pi 5 + Raspberry Pi AI Kit (Hailo-8L NPU)**

**Justification:**  
This is explicitly an **Edge-AI** system: all video processing and inference run **on-device** for low latency, privacy, and zero cloud dependency. Compared to MCU-only TinyML, the Pi 5 + AI Kit provides enough compute for real-time **OpenCV** and optional **INT8 CNN/segmentation** while remaining compact and low-power relative to laptop/GPU solutions. A camera workload (bubble sizing, velocity, void fraction) benefits from this NPU headroom.

---

## 3. Problem Definition
Design an **on-device vision system** that **detects, sizes, and tracks air bubbles** in mineral-oil MTB brake lines/syringes during bleeding. The AI-hardware challenge is to meet **real-time latency** and **low power** with a small NPU via **model compression**, **INT8 deployment**, and **sensor/lighting co-design**, producing robust results under vibration and ambient-light variation.

---

## 4. Technical Objectives
- **Sizing accuracy:** Diameter error ≤ **±0.05 mm** for bubbles ≥ **0.30 mm** (calibrated px→mm).  
- **Detection F1:** **≥ 0.97** for bubbles ≥ **0.30 mm** at **60–120 fps** in clear mineral oil.  
- **Latency:** Capture → decision **≤ 50 ms** median; overlay **≥ 30 fps**.  
- **Void fraction:** Mean absolute error **≤ 0.02** (1-s window).  
- **Robustness:** Maintain **F1 ≥ 0.90** under moderate vibration and lighting variation.

---

## 5. Methodology
### Hardware Setup
- **Compute:** Raspberry Pi 5 + **Raspberry Pi AI Kit (Hailo-8L)** — **Edge-AI** inference.  
- **Camera:** **Global-shutter monochrome** (e.g., OV9281, 1 MP @ 120 fps) with M12 lens (6–12 mm).  
- **Illumination (no backlight required):**  
  - **Dark-field side lighting** using a compact **visible-light LED ring/arc** at a grazing angle (30–60°) to the tube so bubbles scatter light toward the camera while specular glare is minimized.  
  - **Polarization control:** Linear polarizer on lens; optional second polarizer on LEDs for cross-polarization to suppress glare.  
  - **Backdrop:** **Matte-black** shroud/tunnel behind the tube to kill background clutter (still “normal visible light”).  
- **Fixture:** 3D-printed U-clamp that holds tube between camera and side LEDs; adjustable lens-to-tube distance; vibration-damping feet.  
- **Actuation/Sync (optional):** GPIO timestamps from pump/actuator to correlate bubble bursts with stroke phases.  

### Software Tools & Pipeline
- **Capture:** Grayscale ROI at 60–120 fps; **manual** exposure/ISO/WB for stability.  
- **Classic CV (baseline):**  
  1) Background modeling (running median)  
  2) Light blur → **background subtraction**  
  3) **Adaptive/OTSU threshold** + morphology (open/close)  
  4) **Connected components** → per-bubble area, circularity, centroid → diameter (px→mm via calibration)  
  5) Simple tracker (nearest-centroid/Kalman) → **velocity** (mm/s)  
  6) Per-second metrics: **count (Hz), mean/median diameter (mm), velocity (mm/s), void fraction**, system **state** (IDLE → PRIMING → BURST → STABLE)  
- **Optional ML (robustness path):** Lightweight **INT8 UNet-lite** (e.g., 256×256) compiled for Hailo; use the NN mask in place of thresholding under difficult lighting/fluids.  
- **Calibration:** Single mm-scale target in the tube plane for px→mm; store once per rig.  
- **Metrics & Telemetry:** On-screen overlay + CSV/JSON stream  
  ```json
  {"t": ..., "count_hz": ..., "mean_d_mm": ..., "v_mm_s": ..., "void_frac": ..., "state": "..."}


## 6. Expected Deliverables
- Working Demo: Live video overlay showing bubble detections, sizes, velocity in real-time.
- GitHub repository: OpenCV pipeline, optional Haiolo model + compile scripts, fixture CAD, calibration notes
- Dataset: Labeled clips/masks (subset) + metrics CSVs with a README.
- Documentation: BOM, assembly/lighting guide, latency & power benchmarks, model card.
- Presentations: Midterm slides; final report + demo video; archived repo in /docs.

## 7. Team Responsibilities
List each member’s main role.

| Name | Role | Responsibilities |
|------|------|------------------|
| Landon Campbell | Team Lead | Planning, repo/docs, integration, risk & compliance, performance profiling |
| Thomas Keyes | Hardware | Camera/lens selection, lighting & polarization, clamp/shroud design, calibration |
| Tiger Zhang | Software | CV pipeline, optional UNet-lite training & INT8 deploy, test plans, benchmarking |

## 8. Timeline and Milestones
Provide expected milestones:

| Week | Milestone | Deliverable |
|------|------------|-------------|
| 2 | Proposal | PDF + GitHub submission (/docs + issues) |
| 3 | Initial Rig Assembly | CAD Clamp + side-LEDs + camera connection; labeled clips
| 4 | Midterm presentation | Slides + classic CV baseline metrics |
| 5 | INT8 model on Hailo | Segmentation mask → metrics; latency/power numbers
| 6 | Integration & testing | Vibration/light tests; calibration drift report |
| Dec. 18 | Final presentation | Report, demo, GitHub archive |

## 9. Resources Required
- (IN LAB) Edge-AI compute: Raspberry Pi 5 + Raspberry Pi AI Kit (Hailo-8L).
- (MAY NEED TO PURCHASE) Imaging: Global-shutter mono camera (OV9281-class), M12 lens (6–12 mm), visible LED ring/arc, linear polarizer(s), matte-black shroud.
- (CAN 3D PRINT) Mechanical: 3D-printed clamp/baffles; small rail/tripod; vibration-damping feet.
- (ALREADY PERSONALLY OWN) Fluids rig: Mineral oil, peristaltic/gear pump, micro-nozzles for controlled bubbles, basic flow meter, clear tube/syringe.
- (AVAILABLE ONLINE) Software: OpenCV (C++/Python), HailoRT toolchain (if NN used), Python notebooks for analysis, CSV logging scripts.

## 10. References
## 10. References

- Raspberry Pi — **AI Kit & AI HAT+ (Hailo-8L) setup guide**. https://www.raspberrypi.com/documentation/computers/ai.html

- Raspberry Pi — **AI Kit overview** (what’s in the kit, how it connects to Pi 5). https://www.raspberrypi.com/documentation/accessories/ai-kit.html

- Raspberry Pi — **Camera software & controls** (libcamera, manual exposure/WB). https://www.raspberrypi.com/documentation/computers/camera_software.html

- Omnivision **OV9281** (1 MP global-shutter mono) — sensor overview. https://www.ovt.com/products/ov9281/

- OpenCV — **BackgroundSubtractorMOG2** (foreground mask from video). https://docs.opencv.org/3.4/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html

- OpenCV — **connectedComponentsWithStats** (blob sizing, centroids). https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html

- Lighting technique primer — **Bright-field vs. Dark-field** (why side lighting helps bubbles pop). https://advancedillumination.com/lighting-education/bright-field-dark-field-lighting/

- Bubble imaging (open access) — **In-situ measurements of void fraction & bubble size** (methods + image analysis). https://pmc.ncbi.nlm.nih.gov/articles/PMC9873772/

