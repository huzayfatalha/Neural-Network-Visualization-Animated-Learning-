# Neural Network Visualization (Animated Learning)

A 2D OpenGL/freeglut project that visualizes forward signal flow in a neural network using manually implemented graphics algorithms.

## Project Overview

This project presents an animated neural network with three stages:
- Input layer (left)
- Hidden layer (middle)
- Output layer (right)

Neurons are rendered as circles, connections are drawn as lines, and moving signal particles travel from input to hidden to output neurons. The animation simulates learning-style activation propagation in real time.

## Key Features

- Manual graphics primitives usage:
  - Points
  - Lines
  - Circles
  - Polygons
- Manual graphics algorithms (no built-in line/circle primitives):
  - DDA Line Drawing Algorithm
  - Midpoint Circle Algorithm
- 2D transformations:
  - Translation (signal movement)
  - Scaling (activation pulse on neurons)
  - Rotation (signal particle visual effect)
- Real-time animation loop at approximately 60 FPS using `glutTimerFunc`
- Interactive controls:
  - Start/Stop animation
  - Increase/Decrease signal speed
  - Reset signal flow
  - Optional mouse-triggered signal burst

## Algorithms Used

### 1. DDA Line Drawing
Connections between neurons are drawn with DDA:
- Compute `dx`, `dy`
- Use `steps = max(|dx|, |dy|)`
- Increment x and y by constant fractions each step
- Plot each point with `GL_POINTS`

### 2. Midpoint Circle Drawing
Neurons are drawn with the Midpoint Circle algorithm:
- Start at `(x=0, y=r)`
- Update decision parameter each iteration
- Plot 8-way symmetric points
- Repeat until `x >= y`

## Project Structure

- `main.cpp`: Window setup, scene rendering, network layout, animation, transformations, and interaction callbacks
- `algorithms.h`: Manual pixel plotting, DDA line drawing, and midpoint circle drawing
- `animation.h`: Signal data structure used for animated propagation

## Build Requirements

- Windows + g++
- OpenGL
- freeglut

## How to Compile

Run from the project directory:

```powershell
g++ main.cpp -o neural_network_vis -IC:/msys64/ucrt64/include -LC:/msys64/ucrt64/lib -lfreeglut -lopengl32 -lglu32
```

## How to Run

```powershell
./neural_network_vis.exe
```

## Controls

- `P`: Pause/Resume animation
- `+`: Increase signal speed
- `-`: Decrease signal speed
- `R`: Reset signals
- `Esc`: Exit
- `Left Mouse Click` near an input neuron: Emit a small burst of signals

## Screenshots

Add your screenshots here before final submission.

### Suggested screenshots

1. Full network view (idle animation state)
2. Active signal flow from input to hidden
3. Active signal flow reaching output neurons
4. Pause/resume or speed control demonstration

Example markdown format:

```md
![Main View](screenshots/main-view.png)
![Signal Flow](screenshots/signal-flow.png)
![Output Activation](screenshots/output-activation.png)
```

## Learning Outcomes

This project demonstrates:
- Practical implementation of classic computer graphics algorithms
- Real-time animation pipeline in OpenGL/freeglut
- Use of transformation matrices for visual behavior
- Modular C++ project organization suitable for university submission
