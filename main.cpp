// main.cpp
// Advanced 2D neural network visualization with manual algorithms and interactive demo features.

#include <GL/freeglut.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <deque>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "algorithms.h"
#include "animation.h"

const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;
const int FRAME_TIME_MS = 16;

enum LayerId { INPUT_LAYER = 0, HIDDEN1_LAYER = 1, HIDDEN2_LAYER = 2, OUTPUT_LAYER = 3 };
enum PresetId { PRESET_A = 1, PRESET_B = 2, PRESET_C = 3 };

struct Neuron {
    float x;
    float y;
    int radius;
    float r;
    float g;
    float b;
};

struct ConnectionEdge {
    int fromLayer;
    int fromIndex;
    int toLayer;
    int toIndex;
    float weight;
};

struct PresetConfig {
    int inputCount;
    int hidden1Count;
    int hidden2Count;
    int outputCount;
    bool useSecondHidden;
};

std::vector<Neuron> inputLayer;
std::vector<Neuron> hiddenLayer1;
std::vector<Neuron> hiddenLayer2;
std::vector<Neuron> outputLayer;

std::vector<float> inputActivation;
std::vector<float> hidden1Activation;
std::vector<float> hidden2Activation;
std::vector<float> outputActivation;

std::vector<float> inputValues;
std::vector<float> hidden1Values;
std::vector<float> hidden2Values;
std::vector<float> outputValues;

std::vector<ConnectionEdge> forwardEdges;
std::vector<ConnectionEdge> feedbackEdges;

std::vector<Signal> forwardSignals;
std::vector<Signal> feedbackSignals;

std::deque<float> outputHistory;

PresetId currentPreset = PRESET_A;
PresetConfig currentConfig = {4, 5, 0, 3, false};

bool animationRunning = true;
bool demoMode = false;
bool presentationMode = false;
bool showFeedback = true;
bool slowMotion = false;
bool screenshotRequested = false;

float signalSpeedValue = 0.015f;
size_t nextForwardSignalIndex = 0;
size_t nextFeedbackSignalIndex = 0;
int frameCounter = 0;

int fpsValue = 0;
int fpsFrameCount = 0;
int fpsLastTimeMs = 0;

std::mt19937 rng(42);

float clamp01(float value) {
    return std::max(0.0f, std::min(1.0f, value));
}

float sigmoidApprox(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

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
        layer.push_back({x, topY + spacing * static_cast<float>(i), radius, r, g, b});
    }

    return layer;
}

std::vector<Neuron>& getLayerById(int layerId) {
    if (layerId == INPUT_LAYER) {
        return inputLayer;
    }
    if (layerId == HIDDEN1_LAYER) {
        return hiddenLayer1;
    }
    if (layerId == HIDDEN2_LAYER) {
        return hiddenLayer2;
    }
    return outputLayer;
}

std::vector<float>& getActivationById(int layerId) {
    if (layerId == INPUT_LAYER) {
        return inputActivation;
    }
    if (layerId == HIDDEN1_LAYER) {
        return hidden1Activation;
    }
    if (layerId == HIDDEN2_LAYER) {
        return hidden2Activation;
    }
    return outputActivation;
}

std::vector<float>& getValuesById(int layerId) {
    if (layerId == INPUT_LAYER) {
        return inputValues;
    }
    if (layerId == HIDDEN1_LAYER) {
        return hidden1Values;
    }
    if (layerId == HIDDEN2_LAYER) {
        return hidden2Values;
    }
    return outputValues;
}

const char* getPresetName(PresetId preset) {
    if (preset == PRESET_A) {
        return "Preset A (4-5-3)";
    }
    if (preset == PRESET_B) {
        return "Preset B (6-8-4)";
    }
    return "Preset C (3-6-6-2)";
}

void setPresetConfig(PresetId preset) {
    currentPreset = preset;
    if (preset == PRESET_A) {
        currentConfig = {4, 5, 0, 3, false};
    } else if (preset == PRESET_B) {
        currentConfig = {6, 8, 0, 4, false};
    } else {
        currentConfig = {3, 6, 6, 2, true};
    }
}

void connectDense(int fromLayer, int toLayer, std::vector<ConnectionEdge>& targetEdges,
                  std::uniform_real_distribution<float>& weightDist,
                  std::mt19937& localRng) {
    const std::vector<Neuron>& from = getLayerById(fromLayer);
    const std::vector<Neuron>& to = getLayerById(toLayer);

    for (int i = 0; i < static_cast<int>(from.size()); ++i) {
        for (int j = 0; j < static_cast<int>(to.size()); ++j) {
            targetEdges.push_back({fromLayer, i, toLayer, j, weightDist(localRng)});
        }
    }
}

void initializeNetworkLayout() {
    float xInput = 120.0f;
    float xH1 = 350.0f;
    float xH2 = 520.0f;
    float xOut = currentConfig.useSecondHidden ? 680.0f : 640.0f;

    inputLayer = createLayer(currentConfig.inputCount, xInput, 110.0f, 490.0f, 16, 0.15f, 0.45f, 1.00f);
    hiddenLayer1 = createLayer(currentConfig.hidden1Count, xH1, 90.0f, 510.0f, 16, 1.00f, 0.85f, 0.15f);

    if (currentConfig.useSecondHidden) {
        hiddenLayer2 = createLayer(currentConfig.hidden2Count, xH2, 120.0f, 480.0f, 16, 1.00f, 0.85f, 0.15f);
    } else {
        hiddenLayer2.clear();
    }

    outputLayer = createLayer(currentConfig.outputCount, xOut, 170.0f, 430.0f, 16, 1.00f, 0.20f, 0.20f);

    inputActivation.assign(inputLayer.size(), 0.0f);
    hidden1Activation.assign(hiddenLayer1.size(), 0.0f);
    hidden2Activation.assign(hiddenLayer2.size(), 0.0f);
    outputActivation.assign(outputLayer.size(), 0.0f);

    inputValues.assign(inputLayer.size(), 0.5f);
    hidden1Values.assign(hiddenLayer1.size(), 0.0f);
    hidden2Values.assign(hiddenLayer2.size(), 0.0f);
    outputValues.assign(outputLayer.size(), 0.0f);

    forwardEdges.clear();
    feedbackEdges.clear();

    std::mt19937 localRng(100 + static_cast<int>(currentPreset));
    std::uniform_real_distribution<float> weightDist(-1.0f, 1.0f);

    connectDense(INPUT_LAYER, HIDDEN1_LAYER, forwardEdges, weightDist, localRng);
    if (currentConfig.useSecondHidden) {
        connectDense(HIDDEN1_LAYER, HIDDEN2_LAYER, forwardEdges, weightDist, localRng);
        connectDense(HIDDEN2_LAYER, OUTPUT_LAYER, forwardEdges, weightDist, localRng);
    } else {
        connectDense(HIDDEN1_LAYER, OUTPUT_LAYER, forwardEdges, weightDist, localRng);
    }

    for (const ConnectionEdge& edge : forwardEdges) {
        feedbackEdges.push_back({edge.toLayer, edge.toIndex, edge.fromLayer, edge.fromIndex, edge.weight});
    }
}

void applySignalSpeedToAll(float speed) {
    for (Signal& signal : forwardSignals) {
        signal.speed = speed;
    }
    for (Signal& signal : feedbackSignals) {
        signal.speed = speed * 0.9f;
    }
}

Signal makeSignalFromEdge(const ConnectionEdge& edge, bool active, float speed) {
    const Neuron& source = getLayerById(edge.fromLayer)[edge.fromIndex];
    const Neuron& target = getLayerById(edge.toLayer)[edge.toIndex];

    Signal signal = {
        source.x,
        source.y,
        target.x,
        target.y,
        source.x,
        source.y,
        0.0f,
        speed,
        edge.fromLayer,
        edge.fromIndex,
        edge.toLayer,
        edge.toIndex,
        0.0f,
        active
    };
    return signal;
}

void initializeSignals() {
    forwardSignals.clear();
    feedbackSignals.clear();

    for (const ConnectionEdge& edge : forwardEdges) {
        forwardSignals.push_back(makeSignalFromEdge(edge, false, signalSpeedValue));
    }
    for (const ConnectionEdge& edge : feedbackEdges) {
        feedbackSignals.push_back(makeSignalFromEdge(edge, false, signalSpeedValue * 0.9f));
    }

    nextForwardSignalIndex = 0;
    nextFeedbackSignalIndex = 0;
    frameCounter = 0;
    outputHistory.clear();

    if (!forwardSignals.empty()) {
        forwardSignals[0].active = true;
    }
}

float findEdgeWeight(const std::vector<ConnectionEdge>& edges, int fromLayer, int fromIndex,
                     int toLayer, int toIndex) {
    for (const ConnectionEdge& edge : edges) {
        if (edge.fromLayer == fromLayer && edge.fromIndex == fromIndex &&
            edge.toLayer == toLayer && edge.toIndex == toIndex) {
            return edge.weight;
        }
    }
    return 0.0f;
}

void updateNeuronValueOnArrival(const Signal& signal, bool feedback) {
    std::vector<float>& sourceValues = getValuesById(signal.sourceLayer);
    std::vector<float>& targetValues = getValuesById(signal.targetLayer);

    if (signal.sourceIndex < 0 || signal.sourceIndex >= static_cast<int>(sourceValues.size()) ||
        signal.targetIndex < 0 || signal.targetIndex >= static_cast<int>(targetValues.size())) {
        return;
    }

    float weight = feedback
                       ? findEdgeWeight(feedbackEdges, signal.sourceLayer, signal.sourceIndex,
                                        signal.targetLayer, signal.targetIndex)
                       : findEdgeWeight(forwardEdges, signal.sourceLayer, signal.sourceIndex,
                                        signal.targetLayer, signal.targetIndex);

    float sourceValue = sourceValues[signal.sourceIndex];
    float currentTarget = targetValues[signal.targetIndex];

    if (!feedback) {
        float blended = 0.72f * currentTarget + 0.28f * sigmoidApprox(sourceValue * weight * 2.2f);
        targetValues[signal.targetIndex] = clamp01(blended);
    } else {
        // Visual-only backprop style feedback: gently nudge source layer confidence.
        float correction = 0.12f * std::abs(weight) * (0.5f - sourceValue);
        targetValues[signal.targetIndex] = clamp01(currentTarget + correction);
    }
}

void activateNextForwardSignal() {
    if (forwardSignals.empty()) {
        return;
    }

    Signal& signal = forwardSignals[nextForwardSignalIndex];
    signal.progress = 0.0f;
    signal.currentX = signal.startX;
    signal.currentY = signal.startY;
    signal.rotationAngle = 0.0f;
    signal.active = true;

    std::vector<float>& srcActivation = getActivationById(signal.sourceLayer);
    if (signal.sourceIndex >= 0 && signal.sourceIndex < static_cast<int>(srcActivation.size())) {
        srcActivation[signal.sourceIndex] = 1.0f;
    }

    if (signal.sourceLayer == INPUT_LAYER) {
        float phase = static_cast<float>(frameCounter) * 0.085f + signal.sourceIndex * 0.7f;
        inputValues[signal.sourceIndex] = 0.5f + 0.45f * std::sin(phase);
        inputValues[signal.sourceIndex] = clamp01(inputValues[signal.sourceIndex]);
    }

    nextForwardSignalIndex = (nextForwardSignalIndex + 1) % forwardSignals.size();
}

void activateNextFeedbackSignal() {
    if (!showFeedback || feedbackSignals.empty()) {
        return;
    }

    Signal& signal = feedbackSignals[nextFeedbackSignalIndex];
    signal.progress = 0.0f;
    signal.currentX = signal.startX;
    signal.currentY = signal.startY;
    signal.rotationAngle = 0.0f;
    signal.active = true;

    nextFeedbackSignalIndex = (nextFeedbackSignalIndex + 1) % feedbackSignals.size();
}

void updateSignalVector(std::vector<Signal>& signalVector, bool feedbackPass) {
    for (Signal& signal : signalVector) {
        if (!signal.active) {
            continue;
        }

        signal.progress += signal.speed;
        signal.rotationAngle += feedbackPass ? -11.0f : 14.0f;
        if (signal.rotationAngle >= 360.0f || signal.rotationAngle <= -360.0f) {
            signal.rotationAngle = 0.0f;
        }

        if (signal.progress >= 1.0f) {
            signal.progress = 1.0f;
            signal.currentX = signal.targetX;
            signal.currentY = signal.targetY;
            signal.active = false;

            std::vector<float>& targetActivation = getActivationById(signal.targetLayer);
            if (signal.targetIndex >= 0 && signal.targetIndex < static_cast<int>(targetActivation.size())) {
                targetActivation[signal.targetIndex] = feedbackPass ? 0.6f : 1.0f;
            }

            updateNeuronValueOnArrival(signal, feedbackPass);
            continue;
        }

        signal.currentX = signal.startX + (signal.targetX - signal.startX) * signal.progress;
        signal.currentY = signal.startY + (signal.targetY - signal.startY) * signal.progress;
    }
}

float averageOutputValue() {
    if (outputValues.empty()) {
        return 0.0f;
    }

    float sum = 0.0f;
    for (float value : outputValues) {
        sum += value;
    }
    return sum / static_cast<float>(outputValues.size());
}

void updateAnimation() {
    int forwardSpawnInterval = slowMotion ? 12 : 5;
    int feedbackSpawnInterval = slowMotion ? 30 : 14;

    if (frameCounter % forwardSpawnInterval == 0) {
        activateNextForwardSignal();
    }

    if (showFeedback && frameCounter % feedbackSpawnInterval == 0) {
        activateNextFeedbackSignal();
    }

    if (demoMode && frameCounter % 300 == 0) {
        for (int i = 0; i < 10; ++i) {
            activateNextForwardSignal();
        }
    }

    for (float& value : inputActivation) {
        value = std::max(0.0f, value - 0.032f);
    }
    for (float& value : hidden1Activation) {
        value = std::max(0.0f, value - 0.032f);
    }
    for (float& value : hidden2Activation) {
        value = std::max(0.0f, value - 0.032f);
    }
    for (float& value : outputActivation) {
        value = std::max(0.0f, value - 0.032f);
    }

    updateSignalVector(forwardSignals, false);
    updateSignalVector(feedbackSignals, true);

    outputHistory.push_back(averageOutputValue());
    if (outputHistory.size() > 180) {
        outputHistory.pop_front();
    }

    ++frameCounter;
}

void saveScreenshotPPM(const std::string& filePath) {
    std::vector<unsigned char> pixels(WINDOW_WIDTH * WINDOW_HEIGHT * 3);

    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadBuffer(GL_BACK);
    glReadPixels(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

    std::ofstream out(filePath, std::ios::binary);
    if (!out) {
        return;
    }

    out << "P6\n" << WINDOW_WIDTH << " " << WINDOW_HEIGHT << "\n255\n";

    // Flip vertically because OpenGL reads from bottom-left origin.
    for (int y = WINDOW_HEIGHT - 1; y >= 0; --y) {
        const unsigned char* row = &pixels[y * WINDOW_WIDTH * 3];
        out.write(reinterpret_cast<const char*>(row), WINDOW_WIDTH * 3);
    }
}

void requestScreenshot() {
    screenshotRequested = true;
}

void timer(int value) {
    if (animationRunning) {
        updateAnimation();
    }

    int nowMs = glutGet(GLUT_ELAPSED_TIME);
    ++fpsFrameCount;
    if (nowMs - fpsLastTimeMs >= 1000) {
        fpsValue = fpsFrameCount;
        fpsFrameCount = 0;
        fpsLastTimeMs = nowMs;
    }

    glutPostRedisplay();
    glutTimerFunc(FRAME_TIME_MS, timer, value);
}

void drawText(float x, float y, const std::string& text, float r, float g, float b) {
    glColor3f(r, g, b);
    glRasterPos2f(x, y);
    for (char ch : text) {
        glutBitmapCharacter(GLUT_BITMAP_8_BY_13, ch);
    }
}

void drawGraphPanel() {
    if (outputHistory.empty()) {
        return;
    }

    float left = 500.0f;
    float right = 780.0f;
    float bottom = 30.0f;
    float top = 130.0f;

    glColor4f(0.08f, 0.10f, 0.16f, 0.55f);
    glBegin(GL_POLYGON);
    glVertex2f(left, bottom);
    glVertex2f(right, bottom);
    glVertex2f(right, top);
    glVertex2f(left, top);
    glEnd();

    glColor3f(0.45f, 0.55f, 0.75f);
    drawLineDDA(static_cast<int>(left), static_cast<int>(bottom), static_cast<int>(right), static_cast<int>(bottom));
    drawLineDDA(static_cast<int>(left), static_cast<int>(bottom), static_cast<int>(left), static_cast<int>(top));

    float width = right - left - 8.0f;
    float height = top - bottom - 8.0f;
    float step = width / static_cast<float>(std::max(1, static_cast<int>(outputHistory.size() - 1)));

    glColor3f(0.40f, 0.92f, 0.95f);
    float prevX = left + 4.0f;
    float prevY = bottom + 4.0f + height * outputHistory.front();
    size_t idx = 0;
    for (float value : outputHistory) {
        float x = left + 4.0f + step * static_cast<float>(idx);
        float y = bottom + 4.0f + height * value;
        if (idx > 0) {
            drawLineDDA(static_cast<int>(std::round(prevX)), static_cast<int>(std::round(prevY)),
                        static_cast<int>(std::round(x)), static_cast<int>(std::round(y)));
        }
        prevX = x;
        prevY = y;
        ++idx;
    }

    drawText(left + 8.0f, top - 14.0f, "Output Activation History", 0.88f, 0.90f, 0.96f);
}

void keyboard(unsigned char key, int x, int y) {
    (void)x;
    (void)y;

    if (key == 'p' || key == 'P') {
        animationRunning = !animationRunning;
    } else if (key == '+') {
        signalSpeedValue = std::min(0.080f, signalSpeedValue + 0.003f);
        applySignalSpeedToAll(signalSpeedValue);
    } else if (key == '-') {
        signalSpeedValue = std::max(0.004f, signalSpeedValue - 0.003f);
        applySignalSpeedToAll(signalSpeedValue);
    } else if (key == 'r' || key == 'R') {
        initializeSignals();
        applySignalSpeedToAll(signalSpeedValue);
    } else if (key == '1') {
        setPresetConfig(PRESET_A);
        initializeNetworkLayout();
        initializeSignals();
        applySignalSpeedToAll(signalSpeedValue);
    } else if (key == '2') {
        setPresetConfig(PRESET_B);
        initializeNetworkLayout();
        initializeSignals();
        applySignalSpeedToAll(signalSpeedValue);
    } else if (key == '3') {
        setPresetConfig(PRESET_C);
        initializeNetworkLayout();
        initializeSignals();
        applySignalSpeedToAll(signalSpeedValue);
    } else if (key == 'd' || key == 'D') {
        demoMode = !demoMode;
    } else if (key == 'm' || key == 'M') {
        presentationMode = !presentationMode;
    } else if (key == 'l' || key == 'L') {
        slowMotion = !slowMotion;
    } else if (key == 'f' || key == 'F') {
        showFeedback = !showFeedback;
    } else if (key == 's' || key == 'S') {
        requestScreenshot();
    } else if (key == 27) {
        glutLeaveMainLoop();
    }
}

void mouse(int button, int state, int x, int y) {
    if (state != GLUT_DOWN) {
        return;
    }

    float worldX = static_cast<float>(x);
    float worldY = static_cast<float>(WINDOW_HEIGHT - y);

    if (button == GLUT_LEFT_BUTTON) {
        for (const Neuron& neuron : inputLayer) {
            float dx = worldX - neuron.x;
            float dy = worldY - neuron.y;
            float distanceSq = dx * dx + dy * dy;

            if (distanceSq <= static_cast<float>((neuron.radius + 12) * (neuron.radius + 12))) {
                for (int i = 0; i < 6; ++i) {
                    activateNextForwardSignal();
                }
                break;
            }
        }
    }

    if (button == GLUT_RIGHT_BUTTON) {
        for (int i = 0; i < 6; ++i) {
            activateNextFeedbackSignal();
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

void drawWeightedConnection(const Neuron& start, const Neuron& end, float weight, bool feedbackPass) {
    float magnitude = std::abs(weight);
    int thickness = 1 + static_cast<int>(magnitude * 2.2f);

    float dx = end.x - start.x;
    float dy = end.y - start.y;
    float len = std::sqrt(dx * dx + dy * dy);
    if (len < 1e-4f) {
        return;
    }

    float nx = -dy / len;
    float ny = dx / len;

    float baseR = feedbackPass ? 0.92f : (weight >= 0.0f ? 0.38f : 0.80f);
    float baseG = feedbackPass ? 0.42f : (weight >= 0.0f ? 0.68f : 0.36f);
    float baseB = feedbackPass ? 0.30f : (weight >= 0.0f ? 0.95f : 0.30f);
    float intensity = 0.35f + 0.65f * magnitude;

    glColor3f(baseR * intensity, baseG * intensity, baseB * intensity);
    for (int t = -thickness; t <= thickness; ++t) {
        float offX = nx * static_cast<float>(t) * 0.7f;
        float offY = ny * static_cast<float>(t) * 0.7f;
        drawLineDDA(static_cast<int>(std::round(start.x + offX)),
                    static_cast<int>(std::round(start.y + offY)),
                    static_cast<int>(std::round(end.x + offX)),
                    static_cast<int>(std::round(end.y + offY)));
    }
}

void drawNeuronScaled(const Neuron& neuron, float activationLevel, float valueLevel) {
    float scaleFactor = 1.0f + 0.24f * activationLevel;

    glPushMatrix();
    glTranslatef(neuron.x, neuron.y, 0.0f);
    glScalef(scaleFactor, scaleFactor, 1.0f);

    float valueTint = 0.45f + 0.55f * valueLevel;
    glColor3f(neuron.r * valueTint, neuron.g * valueTint, neuron.b * valueTint);
    for (int radius = neuron.radius; radius >= 1; --radius) {
        drawCircleMidpoint(0, 0, radius);
    }

    glColor3f(0.08f, 0.08f, 0.10f);
    drawCircleMidpoint(0, 0, neuron.radius - 2);
    glColor3f(1.0f, 1.0f, 1.0f);
    drawCircleMidpoint(0, 0, std::max(1, static_cast<int>(std::round((neuron.radius - 4) * valueLevel))));
    glPopMatrix();
}

void drawSignal(const Signal& signal, bool feedbackPass) {
    if (!signal.active) {
        return;
    }

    if (feedbackPass) {
        glColor3f(1.00f, 0.55f, 0.25f);
    } else {
        glColor3f(0.78f, 0.95f, 1.00f);
    }

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

void drawOverlay() {
    if (presentationMode) {
        return;
    }

    std::ostringstream status;
    status << "FPS: " << fpsValue
           << " | Speed: " << std::fixed << std::setprecision(3) << signalSpeedValue
           << " | " << getPresetName(currentPreset)
           << " | Running: " << (animationRunning ? "Yes" : "No")
           << " | Demo: " << (demoMode ? "On" : "Off")
           << " | Feedback: " << (showFeedback ? "On" : "Off");

    drawText(14.0f, 580.0f, status.str(), 0.90f, 0.92f, 0.95f);
    drawText(14.0f, 562.0f,
             "Keys: P Pause  +/- Speed  R Reset  1/2/3 Presets  D Demo  M Presentation  L Slow  F Feedback  S Screenshot",
             0.70f, 0.78f, 0.90f);
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();

    float xInput = inputLayer.empty() ? 120.0f : inputLayer[0].x;
    float xH1 = hiddenLayer1.empty() ? 350.0f : hiddenLayer1[0].x;
    float xOut = outputLayer.empty() ? 650.0f : outputLayer[0].x;

    drawLayerPanel(xInput, 120.0f, 0.15f, 0.45f, 1.00f);
    drawLayerPanel(xH1, 130.0f, 1.00f, 0.85f, 0.15f);

    if (currentConfig.useSecondHidden) {
        float xH2 = hiddenLayer2.empty() ? 520.0f : hiddenLayer2[0].x;
        drawLayerPanel(xH2, 130.0f, 1.00f, 0.85f, 0.15f);
    }

    drawLayerPanel(xOut, 120.0f, 1.00f, 0.20f, 0.20f);

    for (const ConnectionEdge& edge : forwardEdges) {
        const Neuron& start = getLayerById(edge.fromLayer)[edge.fromIndex];
        const Neuron& end = getLayerById(edge.toLayer)[edge.toIndex];
        drawWeightedConnection(start, end, edge.weight, false);
    }

    if (showFeedback) {
        for (const ConnectionEdge& edge : feedbackEdges) {
            const Neuron& start = getLayerById(edge.fromLayer)[edge.fromIndex];
            const Neuron& end = getLayerById(edge.toLayer)[edge.toIndex];
            drawWeightedConnection(start, end, edge.weight, true);
        }
    }

    for (size_t i = 0; i < inputLayer.size(); ++i) {
        drawNeuronScaled(inputLayer[i], inputActivation[i], inputValues[i]);
    }
    for (size_t i = 0; i < hiddenLayer1.size(); ++i) {
        drawNeuronScaled(hiddenLayer1[i], hidden1Activation[i], hidden1Values[i]);
    }
    for (size_t i = 0; i < hiddenLayer2.size(); ++i) {
        drawNeuronScaled(hiddenLayer2[i], hidden2Activation[i], hidden2Values[i]);
    }
    for (size_t i = 0; i < outputLayer.size(); ++i) {
        drawNeuronScaled(outputLayer[i], outputActivation[i], outputValues[i]);
    }

    for (const Signal& signal : forwardSignals) {
        drawSignal(signal, false);
    }

    if (showFeedback) {
        for (const Signal& signal : feedbackSignals) {
            drawSignal(signal, true);
        }
    }

    drawGraphPanel();
    drawOverlay();

    if (screenshotRequested) {
        std::ostringstream fileName;
        fileName << "screenshot_" << std::time(nullptr) << ".ppm";
        saveScreenshotPPM(fileName.str());
        screenshotRequested = false;
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
    glutCreateWindow("Neural Network Visualization (Animated Learning) - Advanced");

    glClearColor(0.06f, 0.06f, 0.08f, 1.0f);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, static_cast<double>(WINDOW_WIDTH), 0.0, static_cast<double>(WINDOW_HEIGHT));
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    setPresetConfig(PRESET_A);
    initializeNetworkLayout();
    initializeSignals();
    applySignalSpeedToAll(signalSpeedValue);

    fpsLastTimeMs = glutGet(GLUT_ELAPSED_TIME);

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutTimerFunc(FRAME_TIME_MS, timer, 0);

    glutMainLoop();
    return 0;
}
