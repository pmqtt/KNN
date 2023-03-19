cmake_minimum_required(VERSION 3.16)
project(KNN)

add_compile_options(-O3)
set(CMAKE_CXX_STANDARD 17)


file(GLOB ALL_OBJ "*.cpp" "*.h" )
list(REMOVE_ITEM ALL_OBJ ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

#add_compile_options(-fsanitize=address)
#add_link_options(-fsanitize=address)

add_library(OBJECT_LIB OBJECT ${ALL_OBJ} )

add_executable(KNN main.cpp $<TARGET_OBJECTS:OBJECT_LIB>)
link_directories(/opt/cuda/lib64)
target_link_libraries(KNN OpenCL)


enable_testing()

file(GLOB TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} test/*.cpp)

#add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=all)
#add_link_options(-fsanitize=address,undefined)

foreach(testSrc ${TEST_SRCS})
    #Extract the filename without an extension (NAME_WE)

    get_filename_component(testName ${testSrc} NAME_WE)

    add_executable(${testName} ${testSrc} $<TARGET_OBJECTS:OBJECT_LIB>)

    set_target_properties(${testName} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR}/test/bin)
    link_directories(/opt/cuda/lib64)
    target_link_libraries(${testName} OpenCL)


    #Finally add it to test execution -
    #Notice the WORKING_DIRECTORY and COMMAND
    add_test(NAME ${testName}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test/bin
            COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test/bin/${testName} )

endforeach(testSrc)
