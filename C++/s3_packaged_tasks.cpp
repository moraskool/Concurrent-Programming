#include <iostream>
#include <future>
#include <numeric>
#include <thread>
#include <functional>

using namespace std;

// The function we wish to test run
int add(int x, int y)
{
	this_thread::sleep_for(std::chrono::milliseconds(500));
	cout << "add function runs in : " << std::this_thread::get_id() << std::endl;
	return x + y;
}

// To run asynchronously
void task_thread()
{
	packaged_task<int(int, int)> task_1(add);
	future<int> future_1 = task_1.get_future();

	thread thread_1(std::move(task_1), 5, 6);
	thread_1.detach();

	cout << "task thread - " << future_1.get() << "\n";
}

// To run sequentially in main or any thread
void task_normal()
{
	packaged_task<int(int, int)> task_1(add);
	future<int> future_1 = task_1.get_future();

	// have to call explicitly, unlike async
	task_1(7, 8);  

	std::cout << "task normal - " << future_1.get() << "\n";
}
//int main()
//{
//	task_thread();
//	task_normal();
//	cout << "main thread id : " << std::this_thread::get_id() << std::endl;
//}