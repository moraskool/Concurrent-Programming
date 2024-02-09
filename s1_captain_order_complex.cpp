#include <iostream>
#include <thread>
#include <chrono>
#include <queue>
#include <string>
using namespace std;

queue<string> EngineCrewQueue;
queue<string> CleanerCrewQueue;

// Cleaners
void CleaningDivision(const int& order)
{
    while (true)
    {
        // if no work for cleaners, sleep for 2s
        // else execute order in queue and sleep for 1s
        if (CleanerCrewQueue.empty())
        {
            this_thread::sleep_for(chrono::milliseconds(2000));
        }
        else
        {
            cout << "Cleaners executing order: " << CleanerCrewQueue.front() << endl;;
            CleanerCrewQueue.pop();
            this_thread::sleep_for(chrono::milliseconds(1000));
        }
    }
    printf("Leaving Cleners Division... \n");
}

// Engine Crew
void EngineDivision(const int& order)
{
    while (true)
    {
        // if no work for engine crew, sleep for 2s
        // else execute order in queue and sleep for 1s
        if (EngineCrewQueue.empty())
        {
            this_thread::sleep_for(chrono::milliseconds(2000));
        }
        else
        {
            cout << "Engine Crew executing order: " << EngineCrewQueue.front() << endl;
            EngineCrewQueue.pop();
            this_thread::sleep_for(chrono::milliseconds(1000));
        }
    }
    printf("Leaving Engine Division... \n");
}


// Captain
//int main()
//{
//    bool run = true;
//    int order = 0;
//    
//    // Captain does not have to wait for these orders, so detach
//    thread EngineCrewThread(EngineDivision, std::ref(order));
//    EngineCrewThread.detach();
//
//    thread CleanerCrewThread(CleaningDivision, std::ref(order));
//    CleanerCrewThread.detach();
//
//    while (run)
//    {
//        cout << "Enter 1, 2 or 3. 100 to exit:";
//        cin >> order;
//
//        if (order == 1)
//        {
//            CleanerCrewQueue.push("CLEAN.");    
//        }
//        else if (order == 2)
//        {
//            EngineCrewQueue.push("FULL_STEAM_AHEAD.");
//        }
//        else if (order == 3)
//        {
//            EngineCrewQueue.push("STOP_ENGINE.");
//        }
//        else if (order == 100)
//        {
//            printf("Stop and Exit \n");
//            run = false;
//            break;
//        }
//        else
//        {
//            printf("Invalid order from Captain \n");
//        }
//        //this_thread::sleep_for(chrono::milliseconds(1000));
//    }
//}


