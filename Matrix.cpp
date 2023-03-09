//
// Created by cefour on 08.03.23.
//
#include "Matrix.h"
#include <cmath>
#include <iostream>
#include <random>

std::random_device rd;  // Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
std::uniform_real_distribution<double> distrib(-1.0, 1.0);

VectorArray::VectorArray(const std::size_t dim) : dim_(dim){
    is_allocated_ = false;
    array_ = new double[dim_];
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
    array_ = new double[dim_];
    is_allocated_ = true;
    for(std::size_t i = 0; i < dim_; ++i){
        array_[i] = rhs.array_[i];
    }
}

VectorArray & VectorArray::operator=(const VectorArray & rhs){
    if(&rhs != this) {
        is_allocated_ = false;
        dim_ = rhs.dim_;
        array_ = new double[dim_];
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
    matrix_ = new double*[sz_row_];
    rows_allocated_ = true;
    for(std::size_t i = 0; i < sz_row_; ++i){
        matrix_[i] = new double[sz_col_];
    }
    coloumns_allocated_ = true;
}

MatrixArray::MatrixArray(const MatrixArray & rhs){
    rows_allocated_ = false;
    coloumns_allocated_ = false;
    sz_row_ = rhs.sz_row_;
    sz_col_ = rhs.sz_col_;
    matrix_ = new double*[sz_row_];
    rows_allocated_ = true;
    for(std::size_t i = 0; i < sz_row_; ++i){
        matrix_[i] = new double[sz_col_];
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
        matrix_ = new double *[sz_row_];
        rows_allocated_ = true;
        for (std::size_t i = 0; i < sz_row_; ++i) {
            matrix_[i] = new double[sz_col_];
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




void Vector::add(std::size_t index, double value){
    array_.array_[index] = value;
}

auto Vector::get(std::size_t i)const -> double{
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

void Matrix::add(std::size_t row,std::size_t col, double value){
    matrix_.matrix_[row][col] = value;
}

Matrix Matrix::operator*(const Matrix & rhs){
    Matrix res(size_of_rows_,rhs.size_of_coloumns);
    for(int i = 0; i < size_of_rows_; ++i){
        for(int j = 0; j < rhs.size_of_rows_; ++j){
            double result = 0.0;
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
        double result =0.0;
        //std::cout<<"Res["<<i<<"]:=";
        for(int j = 0; j < size_of_coloumns; ++j){
        //    std::cout<<" "<<matrix_.matrix_[i][j] <<" * " <<rhs.get(j)<<" + ";
            result += matrix_.matrix_[i][j] * rhs.get(j);
        }
      //  std::cout<<" = " <<result<<"\n";
        res.add(i,result);
    }
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
