#pragma once

#include "../computational_geometry/Points.h"

namespace cmpgtry 
{
	// Return the area of the triangle defined by 2 points in XY 2D space,
	// c is to check orientation
	double areaTriangle2D(const Point2d& a, const Point2d& b, const Point2d& c);

	// Return the area of the triangle defined by 2 points in XY 3D space,
	double areaTriangle2D(const Point3d& a, const Point3d& b, const Point3d& c);

	// Return the area of the triangle defined by 3 points in XY 3D space
	double areaTriangle3D(const Point3d& a, const Point3d& b, const Point3d& c);

	// Return an orientation category
	// Return integer indicating relative position of [Point c] related to segment [a b]
	int orientation2D(const Point2d& a, const Point2d& b, const Point2d& c);

	// This is only for 2D in XY plane.
	int orientation3D(const Point3d& a, const Point3d& b, const Point3d& c);
}