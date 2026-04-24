# Neural Network Visualization (Animated Learning)

An advanced 2D OpenGL/freeglut project that visualizes neural signal flow with manual graphics algorithms, runtime presets, weighted connections, activation analytics, and presentation-ready controls.

## Project Overview

This project renders a layered neural network where:
- Neurons are circles
- Connections are manually drawn lines
- Signals are animated particles moving forward and optional feedback directions

The system now supports multiple network architectures and live visual analytics (status overlay + output history graph) for stronger final-year project presentation quality.

## Core Features

- Manual graphics primitives used:
  - Points
  - Lines
  - Circles
  - Polygons
- Manual graphics algorithms:
  - DDA Line Drawing Algorithm
  - Midpoint Circle Algorithm
- 2D transformations:
  - Translation (signal movement)
  - Scaling (activation pulse on neurons)
  - Rotation (signal particle spin effect)
- Real-time timer-driven animation (`glutTimerFunc`, ~60 FPS)
- Runtime architecture presets:
  - Preset A: 4-5-3
  - Preset B: 6-8-4
  - Preset C: 3-6-6-2
- Weighted connection visualization:
  - Color and thickness vary by weight magnitude/sign
- Output activation history graph panel
- Deterministic demo mode for reliable presentation behavior
- Feedback pass visualization (backprop-inspired visual wave)
- Screenshot export hotkey (PPM)
- Presentation mode toggle (clean screen for teacher demo)

## Algorithms Used

### 1. DDA Line Drawing
Connections are drawn manually using DDA:
1. Compute `dx`, `dy`
2. Determine `steps = max(|dx|, |dy|)`
3. Increment by `xStep`, `yStep`
4. Plot each rasterized point via `GL_POINTS`

### 2. Midpoint Circle Drawing
Neurons are drawn with midpoint logic:
1. Initialize `(x=0, y=r)` and decision parameter
2. Plot 8-way symmetric points
3. Update decision parameter per step
4. Continue until `x >= y`

## Project Structure

- `main.cpp`
  - Window setup, animation loop, presets, overlays, controls, screenshot export
- `algorithms.h`
  - `drawPixel`, `drawLineDDA`, `drawCircleMidpoint`
- `animation.h`
  - Signal data structure used in animation flow

## Build Requirements

- Windows
- g++
- OpenGL + freeglut

## Compile

```powershell
g++ main.cpp -o neural_network_vis -IC:/msys64/ucrt64/include -LC:/msys64/ucrt64/lib -lfreeglut -lopengl32 -lglu32
```

## Run

```powershell
.\neural_network_vis.exe
```

## Controls

### Simulation
- `P`: Pause/Resume
- `+`: Increase speed
- `-`: Decrease speed
- `R`: Reset signal streams

### Presets
- `1`: Preset A (4-5-3)
- `2`: Preset B (6-8-4)
- `3`: Preset C (3-6-6-2)

### Demo and Presentation
- `D`: Deterministic demo mode toggle
- `M`: Presentation mode toggle
- `L`: Slow-motion toggle
- `F`: Feedback wave toggle

### Interaction and Output
- `Left Click` near input neurons: burst forward signals
- `Right Click`: burst feedback signals
- `S`: Save screenshot as `.ppm`
- `Esc`: Exit

## Screenshot Notes

Screenshots are saved in the project folder as files like:
- `screenshot_<timestamp>.ppm`

You can convert PPM to PNG/JPG using image tools for report submission.

## Suggested Submission Screenshots

1. Default live view with overlay
2. Preset B network (6-8-4)
3. Preset C network (3-6-6-2)
4. Feedback wave enabled
5. Presentation mode clean scene
6. Output history graph visible

## Learning Outcomes

This project demonstrates:
- Classic raster graphics algorithm implementation
- Structured real-time animation design
- Data-driven visualization of weighted neural flow
- Runtime interaction and feature toggles for professional demos
- Practical software modularity suitable for university final submission
