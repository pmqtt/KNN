#ifndef KNN_MATRIX_H
#define KNN_MATRIX_H


#include <cstdio>
#include <string>
#include <iostream>
#include <CL/cl.h>
#include <cmath>

template<class T>
std::string from_type(T v){
    char *x = (char*)(&v);
    std::string res;
    for(std::size_t i = 0; i < sizeof(T); ++i){
        res += x[i];
    }
    return res;
}

struct VectorArray{
    VectorArray(const std::size_t dim);

    ~VectorArray();

    VectorArray(const VectorArray & rhs);

    VectorArray & operator=(const VectorArray & rhs);

    std::size_t dim_;
    bool is_allocated_;
    float * array_;
};

struct MatrixArray{
    MatrixArray(const std::size_t & sz_row, const std::size_t & sz_col);

    MatrixArray(const MatrixArray & rhs);

    MatrixArray& operator=(const MatrixArray & rhs);

    ~MatrixArray();

    bool rows_allocated_;
    bool coloumns_allocated_;
    std::size_t sz_row_;
    std::size_t sz_col_;
    float ** matrix_;
};

class Vector{
public:
    Vector(std::size_t sz) : dim_(sz),array_(dim_){}

    Vector(const Vector & rhs):array_(rhs.array_){
        dim_ = rhs.dim_;
    }

    static Vector create_one_vector(std::size_t sz){
        Vector res(sz);
        for(std::size_t i = 0; i < sz; ++i){
            res.array_.array_[i] = 1.0;
        }
        return res;
    }

    Vector operator-(const Vector & rhs){
        Vector result(this->dim_);
        for(std::size_t i = 0; i < this->dim_; ++i){
            result.add(i,array_.array_[i] - rhs.array_.array_[i]);
        }
        return result;
    }

    //component multiplication
    Vector operator*(const Vector & rhs){
        Vector result(this->dim_);
        for(std::size_t i = 0; i < this->dim_; ++i){
            result.add(i,array_.array_[i] * rhs.array_.array_[i]);
        }
        return result;
    }
    friend Vector operator-(float scalar, const Vector & v){
        Vector result(v.dim_);
        for(std::size_t i = 0; i < v.dim_; ++i){
            result.add(i,scalar - v.array_.array_[i]);
        }
        return result;

    }

    void add(std::size_t index, float value);

    auto get(std::size_t i)const -> float;

    void print()const;

    void sigmoid();

    void round();

    auto dim()const -> std::size_t{
        return dim_;
    }

    void fingerprint()const{
        std::cout<<"( "<<dim_<< " )";
    }

    auto data() const  -> const float *{
        return array_.array_;
    }

    auto cpy_data()const -> float *{
        float * f = new float [dim_];
        for(int i = 0; i < dim_; ++i){
            f[i] = array_.array_[i];
        }
        return f;
    }
private:
    std::size_t dim_;
    VectorArray array_;
};



class Matrix{
public:
    Matrix(std::size_t row_no, std::size_t col_no) ;

    void add(std::size_t row,std::size_t col, float value);

    void fill_random();

    Matrix operator*(const Matrix & rhs);

    Vector operator*(const Vector & rhs);

    void print()const;
    auto get(std::size_t i, std::size_t j)const -> float{
        return matrix_.matrix_[i][j];
    }

    friend auto operator<<(std::ostream& os, const Matrix & dt) -> std::ostream&;

    void learn(float rate,Matrix &delta){
        for(std::size_t i = 0; i < size_of_rows_; ++i){
            for(std::size_t j = 0; j < size_of_coloumns; ++j){
                matrix_.matrix_[i][j] += rate * delta.matrix_.matrix_[i][j];
            }
        }
    }

    void fingerprint()const{
        std::cout<<"( "<<size_of_rows_<<" , " <<size_of_coloumns<< " )";
    }

    Matrix T()const{
        Matrix res(size_of_coloumns,size_of_rows_);
        for(int i = 0; i < size_of_rows_; i++){
            for(int j = 0; j < size_of_coloumns; j++){
                res.add(j,i,matrix_.matrix_[i][j]);
            }
        }
        return res;
    }
    Vector gpu_mult(cl_kernel kernel,const Vector &rhs);

    auto cpy_data()const->float*{
        float * data= new float[size_of_rows_* size_of_coloumns];
        int k = 0;

        for(int i = 0; i < size_of_rows_; ++i){
            for(int j = 0; j < size_of_coloumns; ++j){
                data[k] = matrix_.matrix_[i][j];
                k++;
            }
        }
        return data;
    }

private:
    std::size_t size_of_rows_;
    std::size_t size_of_coloumns;
    MatrixArray matrix_;


};

auto dot(const Vector &v1, const Vector & v2) -> Matrix;


#endif //KNN_MATRIX_H
