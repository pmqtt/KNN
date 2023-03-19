//
// Created by cefour on 08.03.23.
//
#include "gpu.h"
#include "Matrix.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>

std::random_device rd;  // Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
std::uniform_real_distribution<float> distrib(-1.0, 1.0);

VectorArray::VectorArray(const std::size_t dim) : dim_(dim){
    is_allocated_ = false;
    array_ = new float[dim_];
    is_allocated_ = true;
}

VectorArray::~VectorArray(){
    if(is_allocated_) {
        delete[] array_;
    }
}

VectorArray::VectorArray(const VectorArray & rhs){
    is_allocated_ = false;
    dim_ = rhs.dim_;
    array_ = new float[dim_];
    is_allocated_ = true;
    for(std::size_t i = 0; i < dim_; ++i){
        array_[i] = rhs.array_[i];
    }
}

VectorArray & VectorArray::operator=(const VectorArray & rhs){
    if(&rhs != this) {
        if(is_allocated_){
            delete [] array_;
        }
        is_allocated_ = false;
        dim_ = rhs.dim_;
        array_ = new float[dim_];
        is_allocated_ = true;
        for (std::size_t i = 0; i < dim_; ++i) {
            array_[i] = rhs.array_[i];
        }
    }
    return *this;
}


MatrixArray::MatrixArray(const std::size_t & sz_row, const std::size_t & sz_col){
    rows_allocated_ = false;
    coloumns_allocated_ = false;
    sz_row_ = sz_row;
    sz_col_ = sz_col;
    matrix_ = new float*[sz_row_];
    rows_allocated_ = true;
    for(std::size_t i = 0; i < sz_row_; ++i){
        matrix_[i] = new float[sz_col_];
    }
    coloumns_allocated_ = true;
}

MatrixArray::MatrixArray(const MatrixArray & rhs){
    rows_allocated_ = false;
    coloumns_allocated_ = false;
    sz_row_ = rhs.sz_row_;
    sz_col_ = rhs.sz_col_;
    matrix_ = new float*[sz_row_];
    rows_allocated_ = true;
    for(std::size_t i = 0; i < sz_row_; ++i){
        matrix_[i] = new float[sz_col_];
        for(std::size_t k = 0; k < sz_col_; k++){
            matrix_[i][k] = rhs.matrix_[i][k];
        }
    }
    coloumns_allocated_ = true;
}

MatrixArray& MatrixArray::operator=(const MatrixArray & rhs){
    if( &rhs != this) {
        rows_allocated_ = false;
        coloumns_allocated_ = false;
        sz_row_ = rhs.sz_row_;
        sz_col_ = rhs.sz_col_;
        matrix_ = new float *[sz_row_];
        rows_allocated_ = true;
        for (std::size_t i = 0; i < sz_row_; ++i) {
            matrix_[i] = new float[sz_col_];
            for (std::size_t k = 0; k < sz_col_; k++) {
                matrix_[i][k] = rhs.matrix_[i][k];
            }
        }
        coloumns_allocated_ = true;
    }
    return *this;
}

MatrixArray::~MatrixArray(){
    if( coloumns_allocated_){
        for(std::size_t i = 0; i < sz_row_; ++i){
            delete [] matrix_[i];
        }
    }
    if( rows_allocated_){
        delete [] matrix_;
    }
}




void Vector::add(std::size_t index, float value){
    array_.array_[index] = value;
}

auto Vector::get(std::size_t i)const -> float{
    return array_.array_[i];
}

void Vector::print()const{
    for(std::size_t i = 0; i < dim_; ++i){
        std::cout<<" "<<array_.array_[i]<<" ";
    }
    std::cout<<"\n";
}

void Vector::sigmoid(){
    for(std::size_t i = 0; i < dim_; ++i) {
        array_.array_[i] = 1.0 / (1.0 + exp(-array_.array_[i]));
        if(std::isnan(array_.array_[i])){
            std::cout<<"-array_.array_[i]:="<<-array_.array_[i]<<"\n";
            std::cout<<"exp(-array_.array_[i]):="<<exp(-array_.array_[i])<<"\n";
            exit(-1);
        }
    }
}


void Vector::round(){
    for(std::size_t i = 0; i < dim_; ++i){
        array_.array_[i] = ::round(array_.array_[i]);
    }
}


Matrix::Matrix(std::size_t row_no, std::size_t col_no) :
        size_of_rows_(row_no),
        size_of_coloumns(col_no), matrix_(size_of_rows_,size_of_coloumns){
}

void Matrix::add(std::size_t row,std::size_t col, float value){
    matrix_.matrix_[row][col] = value;
}

Matrix Matrix::operator*(const Matrix & rhs){
    Matrix res(size_of_rows_,rhs.size_of_coloumns);
    for(int i = 0; i < size_of_rows_; ++i){
        for(int j = 0; j < rhs.size_of_rows_; ++j){
            float result = 0.0;
        //    std::cout<<"Res["<<i<<", "<< j<<"]:=";
            for(int k = 0; k < rhs.size_of_coloumns; ++k){
       //         std::cout<<" "<<matrix_.matrix_[i][k] <<" * " <<rhs.matrix_.matrix_[k][i]<<" + ";
                result += matrix_.matrix_[i][k] * rhs.matrix_.matrix_[k][j];
            }
        //    std::cout<<" = " <<result<<"\n";
            res.matrix_.matrix_[i][j] = result;
        }
    }
    return res;
}

Vector Matrix::operator*(const Vector & rhs){
    Vector res(size_of_rows_);
    for(int i = 0; i < size_of_rows_; ++i){
        float result =0.0;
        for(int j = 0; j < size_of_coloumns; ++j){
            result += matrix_.matrix_[i][j] * rhs.get(j);
        }

        res.add(i,result);
    }
    return res;
}

Vector Matrix::gpu_mult(cl_kernel kernel, const Vector &rhs){
    Vector res(size_of_rows_);
    auto a = cpy_data();
    auto matrix = write_memory(a, size_of_rows_ * size_of_coloumns);
    auto vector = write_memory(const_cast<float *>(rhs.data()),rhs.dim());
    auto result = create_memory(const_cast<float *>(res.data()),size_of_rows_);


    auto err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&matrix);
    if(err != CL_SUCCESS){
        std::cout<<__LINE__<<": "<<getErrorString(err)<<"\n";
    }
     clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&vector);
     clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&result);
     int height = size_of_rows_;
     int width = size_of_coloumns;

     clSetKernelArg(kernel,3,sizeof(int),(void*)&height);
     clSetKernelArg(kernel,4,sizeof(int),(void*)&width);

    size_t global_size = height;
    size_t local_size = 1;


    err = clEnqueueNDRangeKernel(get_command_queue(), kernel, 1, NULL, &global_size, &local_size,
                                 0, NULL, NULL);

    if(err != CL_SUCCESS){
        std::cout<<__LINE__<<": "<<getErrorString(err)<<"\n";
    }


    clEnqueueReadBuffer(get_command_queue(), result , CL_TRUE, 0,
                               res.dim()* sizeof(float), const_cast<float *>(res.data()), 0, NULL, NULL);

    clFlush(get_command_queue());
    clFinish(get_command_queue());
    clReleaseMemObject(matrix);
    clReleaseMemObject(vector);
    clReleaseMemObject(result);
    delete [] a;
    return res;
}


void Matrix::fill_random(){
    for(std::size_t i =0; i < size_of_rows_; ++i){
        for(std::size_t j = 0; j < size_of_coloumns; j++){
            matrix_.matrix_[i][j] = distrib(gen);
        }
    }

}


void Matrix::print()const{
    for(int i = 0; i < size_of_rows_; ++i){
        for(int j = 0; j < size_of_coloumns; ++j){
            std::cout<<" "<< matrix_.matrix_[i][j];
        }
        std::cout<<"\n";
    }
}


auto operator<<(std::ostream& os, const Matrix & dt) -> std::ostream&{
    constexpr auto max_precision {std::numeric_limits<float>::digits10 + 3};
    os<<from_type(dt.size_of_rows_)<<from_type(dt.size_of_coloumns);
    for(std::size_t i = 0; i < dt.size_of_rows_; ++i){
        for(std::size_t j = 0; j < dt.size_of_coloumns; j++){
            os << from_type(dt.matrix_.matrix_[i][j]);
        }
    }
    return os;
}

auto dot(const Vector &v1, const Vector & v2) -> Matrix{
    Matrix result(v1.dim(),v2.dim());

    for(std::size_t i = 0; i < v1.dim(); i++){
        for(std::size_t j = 0; j < v2.dim(); j++){
            result.add(i,j,v1.get(i)*v2.get(j));
        }
    }
    return result;
}