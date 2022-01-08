/**
 * @file Timer.cpp
 * @author Ali Emre Keskin (aliemrekskn@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-01-08
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "Timer.h"
#include <chrono>
#include <iostream>

namespace aek
{
    Timer::Timer(std::string mainLabel) : mainLabel_(mainLabel)
    {
        Tic(mainLabel_);
    }
    
    Timer::~Timer() 
    {
        Toc();
    }

    void Timer::Tic(std::string label)
    {
        stack_.push({label, std::chrono::high_resolution_clock::now()});
    }

    void Timer::Toc()
    {
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - stack_.top().second).count();
        std::cout << mainLabel_ << " " << stack_.top().first << " " << duration << std::endl;
        stack_.pop();
    }

} // namespace aek