# FLASH: A Reference-Free “Generate Once, Synthesize Many” Framework for Synthetic Anomaly Generation in Industrial Anomaly Detection

## Project Overview

**FLASH** is a reference-free synthetic anomaly generation framework developed to address the scarcity of real defect samples in industrial anomaly detection.

The project introduces a **“Generate Once, Synthesize Many”** approach. Instead of generating a new anomaly for every image, FLASH generates a limited set of defect instances, extracts and validates them, and stores them in a reusable **category-wise Semantic Defect Bank**. These defects are then synthesized onto different normal host images using **Object Boundary Suppression (OBS), Multi-Resolution Spectral Pyramid (MRSP), adaptive placement, and hybrid blending**.

The framework was developed and evaluated on **MVTec AD 2**, with the generated anomalies evaluated for their ability to support **threshold calibration in the absence of real defect samples**.

**For the complete methodology, algorithms, experimental setup, results, and evaluation, see the [Project Report](FLASH_%20A%20Reference-Free%20“Generate%20Once,%20Synthesize%20Many”%20Framework.pdf).**

---

## Project Report

📄 [**Read the Complete Project Report**](FLASH_%20A%20Reference-Free%20“Generate%20Once,%20Synthesize%20Many”%20Framework.pdf)

The report contains the complete technical details of FLASH, including the five-stage architecture, DiffMask, OBS, MRSP, adaptive synthesis, experimental methodology, MVTec AD 2 evaluation, calibration gap analysis, and computational analysis.

---

## Architecture

![FLASH Architecture](FLASH%20Architecture.png)

**Figure: Five-stage FLASH architecture for reference-free synthetic anomaly generation.**

The FLASH framework separates expensive defect generation from scalable anomaly synthesis through five stages:

### Stage 1: Semantic-Guided Anomaly Generation

A normal image is analyzed by **VLM-1** to identify a plausible category-specific anomaly. The generated anomaly prompt is combined with the configuration file containing dataset context and generation constraints. The image-generation model then produces the initial generated anomaly image.

### Stage 2: Defect Extraction

The normal image and generated anomaly image are processed by **DiffMask**. The method identifies the actual changes introduced by the generated defect and produces a **Defect Mask** and corresponding **Defect Crop**.

### Stage 3: Defect Validation and Storage

The extracted defect crop is passed to **VLM-2** for semantic validation. Valid defects are retained and stored in a **Category-wise Semantic Defect Bank**, allowing the same generated defects to be reused across multiple normal images.

### Stage 4: Object-Aware Localization

A different normal host image is processed using **Object Boundary Suppression (OBS)** to identify the foreground object region. **MRSP** then generates a coherent and size-controllable placement mask within the object region.

### Stage 5: Adaptive Synthesis

A validated defect is retrieved from the Semantic Defect Bank and placed onto the normal host image according to the **MRSP and OBS mask**. The defect is adaptively scaled and positioned, harmonized with the host image, and synthesized using **hybrid blending** to produce the final synthetic anomaly.

---

## Acknowledgements

This project was completed as part of **Google Summer of Code 2026** with the **Intel Benelux Anomalib Team**.

I would like to sincerely thank my mentors, **Ashwin** and **Rajesh**, for their continuous guidance, technical discussions, and mentorship throughout the project. Their guidance helped me develop a stronger understanding of research methodology, problem solving, and practical R&D.

I am grateful to the **OpenVINO Toolkit** and the entire **Intel Benelux Anomalib Team** for providing the opportunity, environment, and support to work on this project.
