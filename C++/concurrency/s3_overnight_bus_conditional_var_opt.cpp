#include <iostream>
#include <thread>
#include <string>
#include <mutex>
#include <chrono>
#include <condition_variable>

using namespace std;

bool arrived = false;
int dist_covered = 0;
int dest_dist = 10;

condition_variable cv;
mutex m;

// Event : Passenger
void keepMoving()
{

	while (true)
	{
		this_thread::sleep_for(chrono::milliseconds(1000));
		dist_covered++;

		//notify waiting thread if event occurs
		if (dist_covered == dest_dist)
		{
			cv.notify_one(); // wake up sleeping threads (Passengers)
		}
	}
}

// Waithing Threads - Bus Driver
void askBusDriverAlertAtRightTime()
{
	//use unique lock for flexibility be able to call lock and unlock 
	unique_lock<mutex> ul(m); 

	// Lambda expr : wait on the condition returns true check after wake-up
	cv.wait(ul, [] {return dist_covered == dest_dist; }); 
	cout << "I am finally here. Distance = " << dist_covered << endl;

}

//int main()
//{
//	thread driver_thread(keepMoving);
//	thread passenger_thread(askBusDriverAlertAtRightTime);
//
//	passenger_thread.join();
//	driver_thread.join();
//
//}