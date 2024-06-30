// ******** Section 1 : Launching Threads   ************ //

#include <iostream>
#include <thread>

void foo()
{
    printf("Hello from foo function - %d \n", std::this_thread::get_id());
}

class callable_class
{
    public:
        void operator()()
        {
            printf("Hello from callable class - %d \n", std::this_thread::get_id());
        }
};

//int main()
//{
//   // launch thread from a function
//   std::thread thread1(foo);      
//
//   // launch thread from a callable object
//   callable_class obj;
//   std::thread thread2(obj);      
//
//   // launch thread from a lambda function
//   std::thread thread3([] 
//   {
//     printf("Hello from lambda function - %d \n", std::this_thread::get_id());
//   });         
//   
//   thread1.join();
//   thread2.join();
//   thread3.join();
//
//   printf("Hello from main function %d - \n", std::this_thread::get_id());
//   
//}