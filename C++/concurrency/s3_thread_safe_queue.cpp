#include <iostream>
#include <thread>
#include <mutex>
#include <string>
#include <memory>
#include <queue>
#include <condition_variable>

using namespace std;

template<typename T>

class ThreadSafeQueue
{
	// define data members
	mutex m;                  // mutually exclusive access to all func
	condition_variable cv;    // waitpop func
	queue<shared_ptr<T>> q;   // wrap queue with function name


	public:

		ThreadSafeQueue() {}
		ThreadSafeQueue(ThreadSafeQueue const& other_q) 
		{
			lock_guard<mutex> lg(other_q.m);
			q = other_q;
		}

		void push(T& value)
		{
			lock_guard<mutex> lg(m);
			q.push(make_shared<T>(value)); // avoid execption scenario from resource mngmt
			cv.notify_one();               // wakes up any waiting threads
		}

		shared_ptr<T> pop()
		{
			lock_guard<mutex> lg(m);

			if (q.empty)
			{
				return shared_ptr<T>();
			}
			else
			{
				shared_ptr<T> ref(q.front());
				q.pop;
				return ref;
			}
		}

		shared_ptr<T> wait_pop()
		{
			unique_lock<mutex> ul(m); // ul is used here for locking flexibility 

			// A thread unlocks the mutex if the condition fails, otherwise acq lock and proceeds
			// notify_one can make one of many threads to wake randomly and acq lock
			// notify_all, wakes up all thread, but only one can proceed
			cv.wait(ul, [this] { return !q.empty(); });
			
			shared_ptr<T> ref = q.front();
			q.pop;
			return ref;

		}

		bool wait_pop(T& ref)
		{
			std::unique_lock<std::mutex> lg(m);
			cv.wait(lg, [this] {
				return !q.empty();
				});

			ref = *(q.front().get());
			q.pop();
			return true;
		}

		bool pop(T& ref)
		{
			std::lock_guard<std::mutex> lg(m);
			if (q.empty())
			{
				return false;
			}
			else
			{
				ref = q.front();
				q.pop();
				return true;
			}
		}

		bool empty()
		{
			lock_guard<mutex> lg(m);
			return q.empty();
		}
		
		size_t size()
		{
			lock_guard<mutex> lg(m);
			return q.size();
		}
};

ThreadSafeQueue<int> q1;

// Thread Functions
void func_1()
{
	int value;
	q1.wait_pop(value);
	cout << value << std::endl;

}

void func_2()
{
	int x = 10;
	this_thread::sleep_for(std::chrono::milliseconds(2000));
	q1.push(x);
}

//int main()
//{
//	thread thread_1(func_1);
//	thread thread_2(func_2);
//
//	thread_1.join();
//	thread_2.join();
//}