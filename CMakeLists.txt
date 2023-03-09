cmake_minimum_required(VERSION 3.24)
project(KNN)

set(CMAKE_CXX_STANDARD 17)

add_executable(KNN main.cpp Matrix.h Matrix.cpp FeedForwardNetwork.h)
