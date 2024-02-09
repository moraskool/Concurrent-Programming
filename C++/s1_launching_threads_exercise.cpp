#include <iostream>
#include <thread>


void test()
{
    printf("Hello from test within functionA %d \n", std::this_thread::get_id());
}

void functionA()
{
   std::thread threadC(test);
   threadC.join();
}

void functionB()
{
    printf("Hello from functionB  %d \n", std::this_thread::get_id());
}

int main()
{
    std::thread threadA(functionA);
    threadA.join();

    std::thread threadB(functionB);
    threadB.join();

    printf("Hello from main function  %d \n ", std::this_thread::get_id());
}