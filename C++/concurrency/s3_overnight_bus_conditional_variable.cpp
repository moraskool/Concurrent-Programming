#include <iostream>
#include <mutex>
#include <thread>
#include <chrono>
#include <string>

using namespace std;

bool have_arrived = false;
int distance_covered = 0;
int destination_dist = 10;


// Event : bus keeps driving and covers travel distance 
bool keepDriving()
{
	while (true)
	{
		this_thread::sleep_for(chrono::milliseconds(1000));
		distance_covered++;
	}

	return false;
}

// The bus driver should be awake and keep driving
// until the destination distance is reached
void keepAwakeAllNight()
{
	while (distance_covered < destination_dist)  // condition check here takes more precessor time
	{
		cout << "Not there yet. Keep driving" << endl;
		this_thread::sleep_for(chrono::milliseconds(1000));
	}

	cout << "Now, I am there. Distance covered = " <<  distance_covered << endl;
}

// Waiting Threads : Passenger naps on alarm
void setAlarmAndNap()
{
	while (distance_covered < destination_dist)
	{
		cout << ".....Taking a Nap" << endl;
		this_thread::sleep_for(chrono::milliseconds(10000));
	}

	cout << "Now, I am there. Distance covered = " << distance_covered << endl;
}


//int main()
//{
//
//	thread keepDriving_thread(keepDriving);
//	thread keepAwakeAllNight_thread(keepAwakeAllNight);
//	thread setAlarmNap_thread(setAlarmAndNap);
//
//	keepAwakeAllNight_thread.join();
//	setAlarmNap_thread.join();
//	keepDriving_thread.join();
//
//	return 0;
//}
