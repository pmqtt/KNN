//
// Created by cefour on 14.03.23.
//
#include <iostream>
#include "../libs/purge/purge.hpp"
#include "../Matrix.h"
#include "../gpu.h"
PURGE_MAIN

#if 0
SIMPLE_TEST_CASE(VECTOR_CONSTRUCTOR){
    Vector vec(10);
    for(std::size_t j = 0; j < 3; j++) {
        for (std::size_t i = 0; i < 10; ++i) {
            vec.add(i, 2.3);
        }
        for (std::size_t i = 0; i < 10; ++i) {
            REQUIRE(vec.get(i) == 2.3);
        }
    }
}

SIMPLE_TEST_CASE(VECTOR_COPY_AND_ASSIGN){
    Vector vec(10);
    for (std::size_t i = 0; i < 10; ++i) {
        vec.add(i, 2.3);
    }
    Vector tmp(vec);
    for (std::size_t i = 0; i < 10; ++i) {
        REQUIRE(tmp.get(i) == 2.3);
    }
    Vector tmp2 = vec;
    for (std::size_t i = 0; i < 10; ++i) {
        REQUIRE(tmp2.get(i) == 2.3);
    }
}

SIMPLE_TEST_CASE(VECTOR_SUB){
    Vector vec(10);
    for (std::size_t i = 0; i < 10; ++i) {
        vec.add(i, 2.3);
    }
    Vector tmp(vec);
    Vector res = vec-tmp;

    for (std::size_t i = 0; i < 10; ++i) {
        REQUIRE(res.get(i) == 0.0);
    }
}

SIMPLE_TEST_CASE(VECTOR_COMPONENT_MULT){
    Vector vec(10);
    for (std::size_t i = 0; i < 10; ++i) {
        vec.add(i, 2.3);
    }
    Vector tmp(vec);
    Vector res = vec*tmp;

    for (std::size_t i = 0; i < 10; ++i) {
        REQUIRE(res.get(i) >= (5.29 -0.01) || res.get(i) >= (5.29 +0.01) );
    }
}
#endif
SIMPLE_TEST_CASE(GPU_MATRIX_VECTOR_MULT){
    init();
    auto kernel = create_kernel_from_file("../../matrix_vec.cl","matrix_vec");
    Matrix m(3,3);
    Vector v(3);

    for(int i = 0; i < 3; ++i){
        for(int j = 0; j < 3; ++j){
            m.add(i,j,i+j);
        }
        v.add(i,i);
    }

    m.print();
    std::cout<<"\n * \n";
    v.print();
    std::cout<<"\n";
    (m*v).print();
    std::cout<<"=========="<<"\n";
    (m.gpu_mult(kernel,v)).print();
}