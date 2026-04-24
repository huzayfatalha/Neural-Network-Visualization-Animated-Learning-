// animation.h
// Animation system declarations (signals, timer updates, activation effects).

#ifndef ANIMATION_H
#define ANIMATION_H

struct Signal {
	float startX;
	float startY;
	float targetX;
	float targetY;
	float currentX;
	float currentY;
	float progress;
	float speed;
	int sourceLayer;
	int sourceIndex;
	int targetLayer;
	int targetIndex;
	float rotationAngle;
	bool active;
};

#endif // ANIMATION_H
