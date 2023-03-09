#ifndef KNN_MATRIX_H
#define KNN_MATRIX_H


#include <cstdio>



struct VectorArray{
    VectorArray(const std::size_t dim);

    ~VectorArray();

    VectorArray(const VectorArray & rhs);

    VectorArray & operator=(const VectorArray & rhs);

    std::size_t dim_;
    bool is_allocated_;
    double * array_;
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
    double ** matrix_;
};

class Vector{
public:
    Vector(std::size_t sz) : dim_(sz),array_(dim_){}

    Vector(const Vector & rhs):array_(rhs.array_){
        dim_ = rhs.dim_;
    }

    void add(std::size_t index, double value);

    auto get(std::size_t i)const -> double;

    void print()const;

    void sigmoid();

private:
    std::size_t dim_;
    VectorArray array_;
};



class Matrix{
public:
    Matrix(std::size_t row_no, std::size_t col_no) ;

    void add(std::size_t row,std::size_t col, double value);

    void fill_random();

    Matrix operator*(const Matrix & rhs);

    Vector operator*(const Vector & rhs);

    void print()const;


private:
    std::size_t size_of_rows_;
    std::size_t size_of_coloumns;
    MatrixArray matrix_;


};




#endif //KNN_MATRIX_H
