#include <iostream>       
#include <functional>     
#include <thread>        
#include <future>       
#include <stdexcept>

void print_int(std::future<int>& fut) {
	std::cout << "waiting for value from print thread \n";
	std::cout << "value: " << fut.get() << '\n';
}

//int main()
//{
//	// thread communication object definition
//	std::promise<int> prom; // sets the value
//	std::future<int> fut = prom.get_future();  // waits for/ gets the promised value
//
//	std::thread print_thread(print_int, std::ref(fut));
//
//	std::this_thread::sleep_for(std::chrono::milliseconds(5000));
//	std::cout << "setting the value in main thread \n";
//	prom.set_value(10);
//
//	print_thread.join();
//}