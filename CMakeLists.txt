cmake_minimum_required(VERSION 3.16)
project(KNN)

#add_compile_options(-O3)
set(CMAKE_CXX_STANDARD 17)
#add_compile_options(-fsanitize=address)
#add_link_options(-fsanitize=address)
add_executable(KNN main.cpp Matrix.h Matrix.cpp FeedForwardNetwork.h)
