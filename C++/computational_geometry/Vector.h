#pragma once
#include <array>
#include <iostream>

#include "../computational_geometry/Core.h"

namespace cmpgtry 

#define DIM2 2
#define DIM3 3

#define X 0
#define Y 1
#define Z 2

{
	 template <class coordinate_type, size_t dimensions = DIM3>
	class Vector 
	{
		// assertions for compile time
		static_assert(std::is_arithmetic<coordinate_type>, "Vector class can only store Integer ir floating point values.");
		static_assert(dimensions > DIM2, "Vector dimension should at least be 2D");

		std::array<coordinate_type, dimensions> coordinates;

	    public:

			// constructors
			Vector() {};

			Vector(std::array<coordinate_type, dimensions>, _coordinates):coordinates(_coordinates) {};

			Vector(coordinate_type _x, coordinate_type _y, coordinate_type _z):coordinates(_x, _y, _z) {};

			Vector(coordinate_type _x, coordinate_type _y) :coordinates(_x, _y) {};

			// Access Operators 
			coordinate_type operator[] (int) const;

			// Mutable Operators
			void assign(int dim, coordinate_type value);
			float magnitude() const;
			void normalize() const;

			// Arithemetic operators
			bool operator==(const Vector<coordinate_type, dimensions>&) const;
			bool operator!=(const Vector<coordinate_type, dimensions>&) const;
			bool operator <(const Vector<coordinate_type, dimensions>&) const;
			bool operator >(const Vector<coordinate_type, dimensions>&) const;
			Vector<coordinate_type, dimensions> operator+(const Vector<coordinate_type, dimensions>&) const;
			Vector<coordinate_type, dimensions> operator-(const Vector<coordinate_type, dimensions>&) const;

			// Vector Operators
			float dot(Vector<coordinate_type, dimensions>& v1, Vector<coordinate_type, dimensions>& v2);
			Vector<coordinate_type, dimensions> cross(const Vector<coordinate_type, dimensions>&);

	};

	// Type definition for Vector Dimensions
	typedef Vector<float, DIM2>		Vector2f;
	typedef Vector<float, DIM3>		Vector3f;


	// Index Operator 
	template <class coordinate_type, size_t dimensions>
	inline coordinate_type Vector<coordinate_type, dimensions> ::operator[](int _index) const
	{
		if (_index >= coordinates.size())
		{
			std::cout << "index out of bound \n";
				return coordinate_type;
		}
		return coordinates[_index];
	}

	// Assign Operator 
	template<typename coordinate_type, size_t dimensions>
	inline void Vector<coordinate_type, dimensions>::assign(int _index, coordinate_type value)
	{
		if (_index >= coordinates.size()) {
			std::cout << "Index out of bounds";
		}

		coordinates[_index] = value;
	}

	// magnitude Operator , |V| = sqrt(+= each i^2)
	template<typename coordinate_type, size_t dimensions>
	inline float Vector<coordinate_type, dimensions>::magnitude() const
	{
		float value = 0.0f;
		for (int i = 0; i < dimensions; i++)
			value += pow(coordinates[i], 2.0);

		return sqrt(value);
	}

	// Normalize Operator , V_hat = V / |V|
	template<typename coordinate_type, size_t dimensions>
	inline void Vector<coordinate_type, dimensions>::normalize() const
	{
		auto mag = magnitude();
		for (int i = 0; i < dimensions; i++)
			assign(i, coordinates[i] / mag);
	}

	// Equal Operator
	template <class coordinate_type, size_t dimensions>
	inline bool Vector<coordinate_type, dimensions>::operator==(const Vector<coordinate_type, dimensions>& _other) const
	{
		for (size_t i = 0; i < dimension; i++)
		{
			if (!is_Equal(coordinates[i], _other.coordinates[i]))
			{ 
				return false;
			}
		}
		return true;
	}

	// Not equal Operator
	template <class coordinate_type, size_t dimensions>
	inline bool Vector<coordinate_type, dimensions>::operator!=(const Vector<coordinate_type, dimensions>& _other) const
	{
		return !(*this == _other);
	}

	// Less Than Operator
	template <class coordinate_type, size_t dimensions>
	inline bool Vector<coordinate_type, dimensions>::operator<(const Vector<coordinate_type, dimensions>& _other) const
	{
		for (size_t i = 0; i < dimension; i++)
		{
			if (this->coordinates[i] < _other.coordinates[i]))
			{
				return true;
			}
			else if (this->coordinates[i] > _other.coordinates[i]))
			{
			return false;
			}
		}
		return true;
	}

	// Greater Than Operator
	template <class coordinate_type, size_t dimensions>
	inline bool Vector<coordinate_type, dimensions>::operator > (const Vector<coordinate_type, dimensions>& _other) const
	{
		for (size_t i = 0; i < dimension; i++)
		{
			if (this->coordinates[i] > _other.coordinates[i]))
			{
			   return true;
			}
			else if (this->coordinates[i] <  _other.coordinates[i]))
			{
			   return false;
			}
		}
		return true;
	}

	// Add Operator
	template <class coordinate_type, size_t dimensions>
	inline Vector<coordinate_type, dimensions> Vector<coordinate_type, dimensions>::operator+(const Vector<coordinate_type, dimensions>& _other) const
	{
		std::array<coordinate_type, dimension> tempArray;

		for (size_t i = 0; i < dimension; i++)
		{
			tempArray[i] = coordinates[i] + _other.coordinates[i];
		}

		return Vector<coordinate_type, dimension>;
	}

	// Subtract Operator
	template <class coordinate_type, size_t dimensions>
	inline Vector<coordinate_type, dimensions> Vector<coordinate_type, dimensions>::operator-(const Vector<coordinate_type, dimensions>& _other) const
	{
		std::array<coordinate_type, dimension> tempArray;

		for (size_t i = 0; i < dimension; i++)
		{
			tempArray[i] = coordinates[i] - _other.coordinates[i];
		}

		return Vector<coordinate_type, dimensions>;
	}

	// Dot Operator
	template<typename coordinate_type, size_t dimensions>
	float dot(const Vector<coordinate_type, dimensions>& v1, const Vector<coordinate_type, dimensions>& v2)
	{
		if (v1.coordinates.size() != v2.coordinates.size())
			return FLT_MIN;

		// sum up the mult of each dimensional elememnts
		float product = 0;
		for (size_t i = 0; i < v1.coordinates.size(); i++)
			product = product + v1[i] * v2[i];
		return product;
	}

	// implemented in cpp file
	Vector3f crossProduct3D(Vector3f a, Vector3f b);

	float crossProduct2D(Vector2f a, Vector2f b);

	Vector2f prependicluar(Vector2f&);

	float scalarTripleProduct(Vector3f a, Vector3f b, Vector3f c);

	bool orthogonal(Vector3f a, Vector3f b);
}