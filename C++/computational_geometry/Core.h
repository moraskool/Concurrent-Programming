#pragma once

#define _USE_MATH_DEFINES

#include<math.h>

#define TOLERANCE 0.0000000001

enum RELATIVE_POS
{
	LEFT, RIGHT, BEHIND, BEYOND, BETWEEN, ORIGIN, DESTINATION
};

static bool isEqualD(double x, double y)
{
	return fabs(x - y) < TOLERANCE;
}
