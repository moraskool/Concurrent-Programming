
#include "Utils.h"

// calculate triangle formed by 3 points
double cmpgtry::areaTriangle2D(const Point2d& a, const Point2d& b, const Point2d& c)
{
	auto AB = b - a;
	auto AC = c - a;

	auto result = crossProduct2D(AB, AC); // area of the parallelogram segment

	return 0.5 * (result);
}

double cmpgtry::areaTriangle2D(const Point3d& a, const Point3d& b, const Point3d& c)
{
	return 0.5 * ((b[X] - a[X]) * (c[Y] - a[Y]) - (c[X] - a[X]) * (b[Y] - a[Y]));
}

int cmpgtry::orientation2D(const Point2d& a, const Point2d& b, const Point2d& c)
{
	double area = areaTriangle2D(a, b, c);

	if (area > 0 && area < TOLERANCE)
		area = 0;

	if (area < 0 && area > TOLERANCE)
		area = 0;

	Vector2f AB = b - a;
	Vector2f AC = c - a;

	if (area > 0.0) return LEFT;
	if (area < 0.0) return RIGHT;
	if (a == c) return ORIGIN;
	if (b == c) return DESTINATION;
	if (AB.magnitude() < AC.magnitude()) return BEYOND;
	if ((AB[X] * AB[X] < 0.0) || (AB[Y] * AC[Y] < 0.0)) return BEHIND;

	return BETWEEN;
}

double cmpgtry::areaTriangle3D(const Point3d& a, const Point3d& b, const Point3d& c)
{
	float x_, y_, z_;

	Vector3f AB = b - a;
	Vector3f AC = c - a;

	x_ = AB[Y] * AC[Z] - AB[Z] * AC[Y];
	y_ = AB[X] * AC[Z] - AB[Z] * AC[X];
	z_ = AB[X] * AC[Y] - AB[Y] * AC[X];

	float sum_of_powers = pow(x_, 2.0) + pow(y_, 2.0) + pow(z_, 2.0);
	float root = sqrtf(sum_of_powers);
	return root / 2;
}

int cmpgtry::orientation3D(const Point3d& a, const Point3d& b, const Point3d& c)
{
	float area = areaTriangle3D(a, b, c);

	if (area > 0 && area < TOLERANCE)
		area = 0;

	if (area < 0 && area > TOLERANCE)
		area = 0;

	Point3d p1 = b - a;
	Point3d p2 = c - a;

	if (area > 0.0) return LEFT;
	if (area < 0.0) return RIGHT;
	if (a == c) return ORIGIN;
	if (b == c) return DESTINATION;
	if (p1.magnitude() < p2.magnitude()) return BEYOND;
	if ((p1[X] * p2[X] < 0.0) || (p1[Y] * p2[Y] < 0.0)) return BEHIND;
	return BETWEEN;
}

