
# Smart Traffic Signal Control System 🚦

A Python-based traffic signal decision-making system leveraging **YOLOv8 vehicle detection**, **multi-object tracking**, and a **Max-Pressure controller** with spillback handling and emergency preemption. Designed for CCTV or drone surveillance video feeds.

---

## Overview

This system automates traffic signal decision-making based on live or recorded video. It detects vehicles, tracks them across frames, computes directional pressures, and selects the optimal traffic light phase. Optional emergency preemption and spillback management ensure realistic and safe operation.

The system is suitable for videos captured from typical 4-legged intersections and can operate on drones or stationary CCTV footage.

---

## Features

- Real-time vehicle detection using **YOLOv8**.
- **Multi-object tracking** with unique IDs and motion trails.
- **Automatic ROI generation** for North (N), South (S), East (E), West (W), and CENTER regions.
- **Max-Pressure controller** for optimal traffic flow:
  - Considers queue lengths in all directions.
  - Handles spillback via CENTER region.
  - Supports min/max green times and clearance intervals.
  - Optional emergency preemption for any direction.
- **Visual Dashboard** displaying:
  - Current phase (NS or EW) with color-coded bars
  - Vehicle counts in N, S, E, W, and CENTER ROIs
  - Pressure values for NS and EW directions
  - Timers for current phase, clearance, and decision epoch
  - Sparkline showing historical total queue
  - Informational captions for decision logic and spillback

---

## Key Components

### Vehicle Detection

- Utilizes **YOLOv8** for detecting vehicles such as cars, trucks, buses, motorcycles, and bicycles.
- Configurable confidence threshold and input image size.
- Works on both CPU and GPU (`cuda`, `mps`) devices.
![WhatsApp Image 2025-09-09 at 13 14 03_bbb6c824](https://github.com/user-attachments/assets/eecde89c-e063-4478-924a-035085140206)


### Region of Interest (ROI) Management

- Automatically generates triangular ROIs for N, S, E, W regions and a rectangular CENTER region.
- The CENTER region is used for **spillback detection** to prevent congestion.
- Custom ROI definitions can be loaded from a JSON file.

### Multi-Object Tracking

- **Track class**: Represents a single object with ID, position, velocity, and motion trail.
- **Tracker class**: Assigns detections to existing tracks using nearest-neighbor matching.
  - New objects spawn new IDs.
  - Tracks lost for a configurable number of frames are removed.
- Provides motion trails and velocity information for visual feedback.

### Max-Pressure Controller

- Uses a **pressure-based control strategy** to choose optimal traffic light phases
- Two main phases:
- Phase 0: NS green, EW red
- Phase 1: EW green, NS red
- Decision logic respects:
- Minimum and maximum green durations
- Clearance intervals between phase switches
- Emergency preemption (for NS or EW)
- Spillback prevention using CENTER region load
- Phase is updated at every **decision epoch** or when maximum green time is reached.

### Dashboard Visualization

- A right-hand panel is added to the video output showing:
- Phase and color-coded bars
- Vehicle counts for all ROIs
- Calculated pressures for NS and EW
- Timers including `t_in_phase`, clearance, and min/max green
- Sparkline showing total queue over time
- Footer caption explaining decision logic
