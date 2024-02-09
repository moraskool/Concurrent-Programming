#include <iostream>
#include <memory>
#include <thread>
#include <mutex>
#include <stdexcept>

using namespace std;

template <typename T>

// optimized thread safe stack 
// to address race condition in
// pop and top

class StackThreadSafe_SharedPtr 
{
	stack <shared_ptr<T>> stk;
	mutex m;

public:
	void push(T element)
	{
		lock_guard<mutex> lg(m);
		stk.push(make_shared(element));
	}

	T& top()
	{
		lock_guard<mutex> lg(m);

		// empty stack throws an exeptions
		if (stk.empty())
		{
			throw runtime_error("stack is empty");
		}
		
		shared_ptr<T> res(stk.top);
		stk.pop();
		return res;	
	}

	 void pop(T& value)
	 {
		lock_guard<mutex> lg(m);

		if (stk.empty())
		{
			throw runtime_error("stack is empty");
		}

		value = *(stk.top().get());  // get value out of stack and assign in refernce
		stk.pop();
	 }

	bool empty()
	{
		lock_guard<mutex> lg(m);
		return stk.empty();
	}

	size_t size()
	{
		lock_guard<mutex> lg(m);
		return stk.size();
	}


};

int main()
{
	return 0;
}