#ifndef __TIMER_H__
#define __TIMER_H__

/**
 * @file Timer.h
 * @author Ali Emre Keskin (aliemrekskn@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-01-08
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <string>
#include <stack>
#include <chrono>

namespace aek
{
    class Timer
    {
    public:
        Timer() = delete;
        Timer(std::string mainLabel);
        ~Timer();
        void Tic(std::string label);
        void Toc();

    private:
        std::stack<std::pair<std::string, std::chrono::_V2::system_clock::time_point>> stack_;
        std::string mainLabel_;
    };
} // namespace aek

#endif // __TIMER_H__