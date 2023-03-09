#ifndef KNN_FEEDFORWARDNETWORK_H
#define KNN_FEEDFORWARDNETWORK_H

#include "Matrix.h"
#include <iostream>
#include <vector>

class FeedForwardNetwork{
public:
    FeedForwardNetwork(std::size_t input_size,std::size_t hidden_layers, std::size_t output_layers) :
                input_size_(input_size),
                hidden_layers_(hidden_layers),
                output_layers_(output_layers),
                output_(output_layers_,input_size){
        for(std::size_t i = 0; i < hidden_layers; ++i){
            Matrix m(input_size_,input_size_);
            m.fill_random();
            network_.push_back(m);
        }

        output_.fill_random();
    }

    auto run(const Vector & input) -> Vector{
        Vector in (input);
        for(std::size_t i = 0; i < hidden_layers_; ++i){
            in = network_[i] * in;
            in.sigmoid();
        }
        auto v = output_ * in;
        v.sigmoid();
        return v;
    }

private:
    std::size_t input_size_;
    std::size_t hidden_layers_;
    std::size_t output_layers_;

    std::vector<Matrix> network_;
    Matrix output_;
};

#endif //KNN_FEEDFORWARDNETWORK_H
