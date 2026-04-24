// main.cpp
// Entry point: OpenGL window setup and render loop wiring.

#include <GL/freeglut.h>

#include <cmath>
#include <vector>

#include "algorithms.h"
#include "animation.h"

const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;
const int FRAME_TIME_MS = 16;

size_t nextSignalIndex = 0;
int frameCounter = 0;
bool animationRunning = true;
float signalSpeedValue = 0.015f;

struct Neuron {
    float x;
    float y;
    int radius;
    float r;
    float g;
    float b;
};

std::vector<Neuron> inputLayer;
std::vector<Neuron> hiddenLayer;
std::vector<Neuron> outputLayer;
std::vector<Signal> signals;
std::vector<float> inputActivation;
std::vector<float> hiddenActivation;
std::vector<float> outputActivation;

std::vector<Neuron> createLayer(int count, float x, float topY, float bottomY, int radius,
                                float r, float g, float b) {
    std::vector<Neuron> layer;
    if (count <= 0) {
        return layer;
    }

    if (count == 1) {
        layer.push_back({x, (topY + bottomY) * 0.5f, radius, r, g, b});
        return layer;
    }

    float spacing = (bottomY - topY) / static_cast<float>(count - 1);
    for (int i = 0; i < count; ++i) {
        layer.push_back({x, topY + i * spacing, radius, r, g, b});
    }

    return layer;
}

void initializeNetworkLayout() {
    // Layer positions from left to right: input, hidden, output.
    inputLayer = createLayer(4, 130.0f, 130.0f, 470.0f, 18, 0.15f, 0.45f, 1.00f);
    hiddenLayer = createLayer(5, 400.0f, 95.0f, 505.0f, 18, 1.00f, 0.85f, 0.15f);
    outputLayer = createLayer(3, 670.0f, 180.0f, 420.0f, 18, 1.00f, 0.20f, 0.20f);

    inputActivation.assign(inputLayer.size(), 0.0f);
    hiddenActivation.assign(hiddenLayer.size(), 0.0f);
    outputActivation.assign(outputLayer.size(), 0.0f);
}

void initializeSignals() {
    signals.clear();

    // Stage 1 routes: input -> hidden.
    for (size_t i = 0; i < inputLayer.size(); ++i) {
        for (size_t j = 0; j < hiddenLayer.size(); ++j) {
            const Neuron& inNeuron = inputLayer[i];
            const Neuron& hidNeuron = hiddenLayer[j];
            Signal signal = {
                inNeuron.x,
                inNeuron.y,
                hidNeuron.x,
                hidNeuron.y,
                inNeuron.x,
                inNeuron.y,
                0.0f,
                signalSpeedValue,
                0,
                static_cast<int>(i),
                1,
                static_cast<int>(j),
                0.0f,
                false
            };
            signals.push_back(signal);
        }
    }

    // Stage 2 routes: hidden -> output.
    for (size_t i = 0; i < hiddenLayer.size(); ++i) {
        for (size_t j = 0; j < outputLayer.size(); ++j) {
            const Neuron& hidNeuron = hiddenLayer[i];
            const Neuron& outNeuron = outputLayer[j];
            Signal signal = {
                hidNeuron.x,
                hidNeuron.y,
                outNeuron.x,
                outNeuron.y,
                hidNeuron.x,
                hidNeuron.y,
                0.0f,
                signalSpeedValue,
                1,
                static_cast<int>(i),
                2,
                static_cast<int>(j),
                0.0f,
                false
            };
            signals.push_back(signal);
        }
    }

    nextSignalIndex = 0;
    frameCounter = 0;
}

void applySignalSpeedToAll(float speed) {
    for (Signal& signal : signals) {
        signal.speed = speed;
    }
}

void activateNextSignal() {
    if (signals.empty()) {
        return;
    }

    Signal& signal = signals[nextSignalIndex];
    signal.progress = 0.0f;
    signal.currentX = signal.startX;
    signal.currentY = signal.startY;
    signal.rotationAngle = 0.0f;
    signal.active = true;

    if (signal.sourceLayer == 0 && signal.sourceIndex >= 0 &&
        signal.sourceIndex < static_cast<int>(inputActivation.size())) {
        inputActivation[signal.sourceIndex] = 1.0f;
    } else if (signal.sourceLayer == 1 && signal.sourceIndex >= 0 &&
               signal.sourceIndex < static_cast<int>(hiddenActivation.size())) {
        hiddenActivation[signal.sourceIndex] = 1.0f;
    }

    nextSignalIndex = (nextSignalIndex + 1) % signals.size();
}

void updateAnimation() {
    // Spawn a new signal wave every few frames for continuous data flow.
    if (frameCounter % 5 == 0) {
        activateNextSignal();
    }

    for (float& value : inputActivation) {
        value = std::max(0.0f, value - 0.035f);
    }
    for (float& value : hiddenActivation) {
        value = std::max(0.0f, value - 0.035f);
    }
    for (float& value : outputActivation) {
        value = std::max(0.0f, value - 0.035f);
    }

    for (Signal& signal : signals) {
        if (!signal.active) {
            continue;
        }

        signal.progress += signal.speed;
        signal.rotationAngle += 14.0f;
        if (signal.rotationAngle >= 360.0f) {
            signal.rotationAngle -= 360.0f;
        }

        if (signal.progress >= 1.0f) {
            signal.progress = 1.0f;
            signal.currentX = signal.targetX;
            signal.currentY = signal.targetY;
            signal.active = false;

            if (signal.targetLayer == 1 && signal.targetIndex >= 0 &&
                signal.targetIndex < static_cast<int>(hiddenActivation.size())) {
                hiddenActivation[signal.targetIndex] = 1.0f;
            } else if (signal.targetLayer == 2 && signal.targetIndex >= 0 &&
                       signal.targetIndex < static_cast<int>(outputActivation.size())) {
                outputActivation[signal.targetIndex] = 1.0f;
            }

            continue;
        }

        // Linear interpolation of signal position between start and target.
        signal.currentX = signal.startX + (signal.targetX - signal.startX) * signal.progress;
        signal.currentY = signal.startY + (signal.targetY - signal.startY) * signal.progress;
    }

    ++frameCounter;
}

void timer(int value) {
    if (animationRunning) {
        updateAnimation();
    }
    glutPostRedisplay();
    glutTimerFunc(FRAME_TIME_MS, timer, value);
}

void keyboard(unsigned char key, int x, int y) {
    (void)x;
    (void)y;

    if (key == 'p' || key == 'P') {
        animationRunning = !animationRunning;
    } else if (key == '+') {
        signalSpeedValue = std::min(0.08f, signalSpeedValue + 0.003f);
        applySignalSpeedToAll(signalSpeedValue);
    } else if (key == '-') {
        signalSpeedValue = std::max(0.004f, signalSpeedValue - 0.003f);
        applySignalSpeedToAll(signalSpeedValue);
    } else if (key == 'r' || key == 'R') {
        initializeSignals();
        applySignalSpeedToAll(signalSpeedValue);
    } else if (key == 27) {
        glutLeaveMainLoop();
    }
}

void mouse(int button, int state, int x, int y) {
    if (button != GLUT_LEFT_BUTTON || state != GLUT_DOWN) {
        return;
    }

    // Convert window coordinates to orthographic world coordinates.
    float worldX = static_cast<float>(x);
    float worldY = static_cast<float>(WINDOW_HEIGHT - y);

    // Optional interaction: click near an input neuron to emit a small burst.
    for (const Neuron& neuron : inputLayer) {
        float dx = worldX - neuron.x;
        float dy = worldY - neuron.y;
        float distanceSq = dx * dx + dy * dy;

        if (distanceSq <= static_cast<float>((neuron.radius + 10) * (neuron.radius + 10))) {
            for (int i = 0; i < 4; ++i) {
                activateNextSignal();
            }
            break;
        }
    }
}

void drawLayerPanel(float centerX, float width, float r, float g, float b) {
    float left = centerX - width * 0.5f;
    float right = centerX + width * 0.5f;

    glColor4f(r, g, b, 0.10f);
    glBegin(GL_POLYGON);
    glVertex2f(left, 40.0f);
    glVertex2f(right, 40.0f);
    glVertex2f(right, 560.0f);
    glVertex2f(left, 560.0f);
    glEnd();
}

void drawConnection(const Neuron& start, const Neuron& end) {
    drawLineDDA(static_cast<int>(std::round(start.x)), static_cast<int>(std::round(start.y)),
                static_cast<int>(std::round(end.x)), static_cast<int>(std::round(end.y)));
}

void drawNeuron(const Neuron& neuron) {
    glColor3f(neuron.r, neuron.g, neuron.b);

    // Draw as transformed local geometry so scaling can pulse activation.
    glPushMatrix();
    glTranslatef(neuron.x, neuron.y, 0.0f);

    // Fill-like effect by drawing multiple concentric midpoint circles.
    for (int radius = neuron.radius; radius >= 1; --radius) {
        drawCircleMidpoint(0, 0, radius);
    }

    glColor3f(0.08f, 0.08f, 0.10f);
    drawCircleMidpoint(0, 0, neuron.radius - 2);
    glPopMatrix();
}

void drawNeuronScaled(const Neuron& neuron, float activationLevel) {
    float scaleFactor = 1.0f + 0.25f * activationLevel;

    glPushMatrix();
    glTranslatef(neuron.x, neuron.y, 0.0f);
    glScalef(scaleFactor, scaleFactor, 1.0f);

    glColor3f(neuron.r, neuron.g, neuron.b);

    // Scale transformation applies to the full neuron body.
    for (int radius = neuron.radius; radius >= 1; --radius) {
        drawCircleMidpoint(0, 0, radius);
    }

    glColor3f(0.08f, 0.08f, 0.10f);
    drawCircleMidpoint(0, 0, neuron.radius - 2);
    glPopMatrix();
}

void drawSignal(const Signal& signal) {
    if (!signal.active) {
        return;
    }

    glColor3f(0.95f, 0.95f, 0.95f);

    // Translation and rotation are applied to local particle coordinates.
    glPushMatrix();
    glTranslatef(signal.currentX, signal.currentY, 0.0f);
    glRotatef(signal.rotationAngle, 0.0f, 0.0f, 1.0f);

    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            drawPixel(dx, dy);
        }
    }
    drawPixel(0, 3);
    drawPixel(0, -3);

    glPopMatrix();
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();

    // Subtle polygon bands make layer groups visually organized.
    drawLayerPanel(130.0f, 120.0f, 0.15f, 0.45f, 1.00f);
    drawLayerPanel(400.0f, 130.0f, 1.00f, 0.85f, 0.15f);
    drawLayerPanel(670.0f, 120.0f, 1.00f, 0.20f, 0.20f);

    // Draw layer-to-layer dense connectivity.
    glColor3f(0.48f, 0.58f, 0.82f);
    for (const Neuron& inNeuron : inputLayer) {
        for (const Neuron& hidNeuron : hiddenLayer) {
            drawConnection(inNeuron, hidNeuron);
        }
    }

    glColor3f(0.82f, 0.62f, 0.42f);
    for (const Neuron& hidNeuron : hiddenLayer) {
        for (const Neuron& outNeuron : outputLayer) {
            drawConnection(hidNeuron, outNeuron);
        }
    }

    // Draw neurons on top of connections.
    for (size_t i = 0; i < inputLayer.size(); ++i) {
        drawNeuronScaled(inputLayer[i], inputActivation[i]);
    }
    for (size_t i = 0; i < hiddenLayer.size(); ++i) {
        drawNeuronScaled(hiddenLayer[i], hiddenActivation[i]);
    }
    for (size_t i = 0; i < outputLayer.size(); ++i) {
        drawNeuronScaled(outputLayer[i], outputActivation[i]);
    }

    for (const Signal& signal : signals) {
        drawSignal(signal);
    }

    glutSwapBuffers();
}

void reshape(int width, int height) {
    glViewport(0, 0, width, height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, static_cast<double>(WINDOW_WIDTH), 0.0, static_cast<double>(WINDOW_HEIGHT));

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

int main() {
    int argc = 1;
    char* argv[1] = {(char*)"NeuralNetworkVis"};

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("Neural Network Visualization (Animated Learning)");

    glClearColor(0.06f, 0.06f, 0.08f, 1.0f);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

    // Initial projection setup for 2D scene coordinates.
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, static_cast<double>(WINDOW_WIDTH), 0.0, static_cast<double>(WINDOW_HEIGHT));
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    initializeNetworkLayout();
    initializeSignals();

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutTimerFunc(FRAME_TIME_MS, timer, 0);

    glutMainLoop();
    return 0;
}
