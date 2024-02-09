#include <iostream>
#include <thread>
#include <stack>
#include <mutex>
#include <memory>

template <typename T>

/*
class SimpleThreadSafeStack
{
	stack<int> s;
	mutex m;

	public:

		// access to all these funtions is mutually exclusive if 
		// use same mutex
		// any func that creates the lg first executes first 
		// others have to wait.

		void push(T element)
		{
			lock_guard<mutex> lg(m);
			s.push_front(element);
		}

		void pop()
		{
			lock_guard<mutex> lg(m);
			s.pop();
		}

		T& top()
		{
			lock_guard<mutex> lg(m);
			return s.top();
		}

		bool empty()
		{
			lock_guard<mutex> lg(m);
			return s.empty();
		}

		size_t size()
		{
			lock_guard<mutex> lg(m);
			return s.size();
		}
};
*/

class ThreadSafeStack
{
	stack<shared_ptr<S>> s;
	mutex m;

public:

	// access to all these funtions is mutually exclusive if 
	// use same mutex
	// any func that creates the lg first executes first 
	// others have to wait.

	void push(T element)
	{
		lock_guard<mutex> lg(m);
		s.push_front(element);
	}

	void pop()
	{
		lock_guard<mutex> lg(m);
		s.pop();
	}

	T& top()
	{
		lock_guard<mutex> lg(m);
		return s.top();
	}

	bool empty()
	{
		lock_guard<mutex> lg(m);
		return s.empty();
	}

	size_t size()
	{
		lock_guard<mutex> lg(m);
		return s.size();
	}
};

int main()
{
	return 0;
}