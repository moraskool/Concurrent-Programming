#include <future>
#include <iostream>
#include <string>
#include <thread>

using namespace std;

int doAddition(int x, int y)
{
	cout << "Printing runs on:" << this_thread::get_id() << endl;
	return x + y;

}
int doSubtraction(int x, int y)
{
	cout << "Printing runs on:" << this_thread::get_id() << endl;
	return x - y;
}

void doPrinting()
{
	cout << "Printing runs on:" << this_thread::get_id() << endl;
}

//
//int main() 
//{
//	cout << "Main thread:" << this_thread::get_id() << endl;
//
//	int x = 80;
//	int y = 50;
//
//	// 3 diff scenarios for async task with lauch params.
//	future<void> f1 = async(launch::async, doPrinting);
//	future<int> f2 = async(launch::deferred, doAddition, x, y);
//	future<int> f3 = async(launch::deferred | launch::async, doSubtraction, x, y);
//
//	f1.get();
//	cout << "Value received using f2 future = " << f2.get() << endl;
//	cout << "Value received using f3 future = " << f3.get() << endl;
//}