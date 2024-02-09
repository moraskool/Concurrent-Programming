#include <iostream>
#include <thread>
#include <list>
#include <mutex>

using namespace std;

class list_wrapper
{
	list <int> my_list;
	mutex m;

	public:
		void add_to_list(int const& l)
		{
			// transform this using lock_guard
			// m.lock();
			// my_list.push_front(l);
			// m.unlock();

			lock_guard<mutex> lg(m);
			my_list.push_front(l);
		}

		void increase_size()
		{
			lock_guard<mutex> lg(m);
			int s = my_list.size();

			cout << "Size of the list is:" << my_list.size() << endl;
		}

		// returns a pointer to the protected list
		list<int>* get_data()
		{
			return &my_list;
		}
};


//int main()
//{
//	list_wrapper *lw;
//	thread thread_1(lw.add_to_list, 4);
//	thread thread_2(lw.add_to_list, 8);
//
//	thread_1.join();
//	thread_2.join();
//}
