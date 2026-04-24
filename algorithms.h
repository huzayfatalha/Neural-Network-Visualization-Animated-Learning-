// algorithms.h
// Manual graphics algorithms (DDA line, Midpoint circle) declarations.

#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#include <GL/freeglut.h>
#include <cmath>

// Draw a single pixel using OpenGL points primitive.
inline void drawPixel(int x, int y) {
	glBegin(GL_POINTS);
	glVertex2i(x, y);
	glEnd();
}

// DDA (Digital Differential Analyzer) line drawing algorithm.
// It increments x and y in small uniform steps and plots rounded points.
inline void drawLineDDA(int x1, int y1, int x2, int y2) {
	int dx = x2 - x1;
	int dy = y2 - y1;

	int steps = std::max(std::abs(dx), std::abs(dy));
	if (steps == 0) {
		drawPixel(x1, y1);
		return;
	}

	float xIncrement = static_cast<float>(dx) / static_cast<float>(steps);
	float yIncrement = static_cast<float>(dy) / static_cast<float>(steps);

	float x = static_cast<float>(x1);
	float y = static_cast<float>(y1);

	for (int i = 0; i <= steps; ++i) {
		drawPixel(static_cast<int>(std::round(x)), static_cast<int>(std::round(y)));
		x += xIncrement;
		y += yIncrement;
	}
}

// Plot all 8 symmetric points for a circle.
inline void plotCircleSymmetry(int cx, int cy, int x, int y) {
	drawPixel(cx + x, cy + y);
	drawPixel(cx - x, cy + y);
	drawPixel(cx + x, cy - y);
	drawPixel(cx - x, cy - y);
	drawPixel(cx + y, cy + x);
	drawPixel(cx - y, cy + x);
	drawPixel(cx + y, cy - x);
	drawPixel(cx - y, cy - x);
}

// Midpoint circle algorithm.
// Uses a decision parameter to choose between East and South-East pixels.
inline void drawCircleMidpoint(int cx, int cy, int radius) {
	int x = 0;
	int y = radius;
	int d = 1 - radius;

	plotCircleSymmetry(cx, cy, x, y);

	while (x < y) {
		++x;

		if (d < 0) {
			d += 2 * x + 1;
		} else {
			--y;
			d += 2 * (x - y) + 1;
		}

		plotCircleSymmetry(cx, cy, x, y);
	}
}

#endif // ALGORITHMS_H
