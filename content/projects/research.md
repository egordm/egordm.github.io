---
title: "Research"
description: "Pre-build surveys and reference notes for projects I am considering building, plus the artifacts that survive when I decide not to build them."
tags:
  - "research"
  - "robotics"
  - "computer-vision"
  - "machine-learning"
---

**Research** is the umbrella project for the structured passes I run over a research-and-engineering landscape before committing to a build. Each pass inventories what is currently shipping, what is publishable but not deployable, what is standardized in the surrounding ecosystem, and what fits on the hardware I actually have. The output is a public survey, a build/no-build decision, and a starting bibliography for anyone considering similar work.

## Why this exists

A lot of "I should build X" intuitions evaporate after thirty hours of literature and ecosystem reading. The Research project is the discipline of doing that reading *in public*: writing the survey down so that the decision is auditable, the references are reusable, and the work is not lost when the conclusion is "do not build".

The decisions themselves are first-class artifacts. A well-defended *no* is more useful than another half-finished library, and a well-defended *yes* makes the resulting library much sharper from day one.

## Series

The surveys are published as the **[[series/research-passes|Research Passes]]** blog series. Each post follows the same shape: industry production state, academic frontier, adjacent fields, standardized engineering layer, edge inference reality, and a short discussion of the most promising libraries and the gaps that remain.

## Current passes

1. **[[blog/2026-05-23-drone-to-satellite-localization-2026|Drone-to-Satellite Visual Geo-Localization (May 2026)]]** - Cross-view localization for UAVs: NVIDIA Isaac ROS, the CVPR 2026 VGGT cluster, MAVLink integration, and what fits on Jetson Orin.
2. **[[blog/2026-05-24-vio-drones-2026|Visual-Inertial Odometry for Drones (May 2026)]]** - VIO for drones: the cuVSLAM squeeze, foundation-3D + IMU research, and the case for two-tier on-drone / off-drone deployment.

More passes follow as new research areas come up.

## Reusable outputs

Independent of whether any single pass becomes a library, the project keeps producing:

- **Annotated bibliographies** of foundation-model + robotics intersections, refreshed at the time of each pass.
- **MAVLink and ROS 2 integration notes** for plugging perception output into existing autopilots.
- **Edge-inference budgets** (Jetson Orin Nano / NX / AGX, M-series Mac) for the model families I evaluate.
- **Calibrated-uncertainty recipes** that apply to any perception module that has to feed an EKF.

If you are starting a similar survey, the [Research Passes series page](/series/research-passes) is the cleanest entry point.
