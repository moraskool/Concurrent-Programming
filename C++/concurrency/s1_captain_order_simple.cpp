#include <iostream>
#include <thread>
#include <chrono>


// Cleaners
void clean()
{
    printf("Cleaning.....\n");
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    printf("Cleaning finished \n");
}

// Engine Crew
void full_speed_ahead()
{
    printf("Increasing ship's speed... \n");
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    printf("Finished increasing ship's speed \n");
}

// EngineCrew
void stop_engine()
{
    printf("Stopping engine.... \n");
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    printf("Stopped \n");
}

// Captain
//int main()
//{
//    bool run = true;
//    while (run)
//    {
//        int order;
//        std::cout << "Enter order : ";
//        std::cin >> order;
//
//        if (order == 1)
//        {
//            // Captain does not have to wait for these orders
//            std::thread clean_thread(clean);
//            clean_thread.detach();
//        }
//        else if (order == 2)
//        {
//            // Captain has to wait for these orders
//            std::thread full_speed_ahead_thread(full_speed_ahead);
//            full_speed_ahead_thread.join();
//        }
//        else if (order == 3)
//        {
//            std::thread stop_engine_thread(stop_engine);
//            stop_engine_thread.join();
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
//
//        printf("Back to Captain's ordrs \n");
//    }
//}