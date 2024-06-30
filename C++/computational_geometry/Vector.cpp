#include "Vector.h"

cmpgtry::Vector3f cmpgtry::crossProduct3D(Vector3f a, Vector3f b)
{
	float x_, y_, z_;
	x_ = a[Y] * b[Z] - b[Y] * a[Z];
	y_ = -(b[Z] * a[X] - a[Z] * b[X]);
	z_ = a[X] * b[Y] - b[X] * a[Y];

	return Vector3f(x_, y_, z_);
}

float cmpgtry::crossProduct2D(Vector2f a, Vector2f b)
{
	return 0.0f;
}

float cmpgtry::scalarTripleProduct(Vector3f a, Vector3f b, Vector3f c)
{
	Vector3f b_cross_c = crossProduct3D(b, c);
	float value = dot(a, b_cross_c);
	return value;
}

bool cmpgtry::orthogonal(Vector3f a, Vector3f b)
{
	float value = dot(a, b);
	return isEqualD(value, 0.0);
}

cmpgtry::Vector2f cmpgtry::prependicluar(Vector2f& vec)
{
	return Vector2f(vec[Y], -vec[X]);
}