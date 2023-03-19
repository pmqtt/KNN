#ifndef KNN_FEEDFORWARDNETWORK_H
#define KNN_FEEDFORWARDNETWORK_H

#include "Matrix.h"
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include "gpu.h"


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
        network_.push_back(output_);
        init();
        kernel = create_kernel_from_file("matrix_vec.cl","matrix_vec");
    }

    auto run(const Vector & input) -> Vector{
        Vector in (input);
        network_outputs_.push_back(in);
        for(std::size_t i = 0; i < hidden_layers_+1; ++i){
            //in = network_[i].gpu_mult(kernel, in );
            in = network_[i] * in;
            in.sigmoid();
            network_outputs_.push_back(in);
        }
        //network_outputs_[network_outputs_.size()-2].print();
        return in;
    }

    void store(const std::string &filename){
        std::ofstream stream;
        stream.open(filename);
        stream<<from_type(input_size_)<<from_type(hidden_layers_)<<from_type(output_layers_);
        for(const auto & x: network_  ){
            stream<<x;
        }
        stream<<output_;
        stream.close();
    }

    template<class T>
    static auto from_byte(const std::string &v) -> T{
        T res;
        char *x = (char*)(&res);
        for(std::size_t i = 0; i < sizeof(T); ++i){
            x[i] = v[i];
        }
        return res;
    }

    template<class T>
    static auto read_type(const std::string & v, std::size_t & pos) -> T{
        std::string buffer;
        for(std::size_t i = 0; i < sizeof(T); ++i ){
            buffer += v[pos+i];
        }
        pos += sizeof(T);
        return from_byte<T>(buffer);
    }

    static FeedForwardNetwork load(const std::string &filename){
        std::ifstream stream;
        stream.open(filename);
        std::string line;
        while(!stream.eof()){
            char c;
            stream.read(&c,1);
            line +=c;
        }

        std::vector<std::string> content;
        std::string item;
        size_t pos = 0;
        size_t isize = read_type<size_t>(line,pos);
        size_t hsize = read_type<size_t>(line,pos);
        size_t osize = read_type<size_t>(line,pos);
        FeedForwardNetwork network(isize,hsize,osize);

        for(int i = 0; i < hsize; ++i){
            size_t rows_count = read_type<size_t>(line,pos);
            size_t cols_count = read_type<size_t>(line,pos);
            for(int j = 0; j < rows_count; j++){
                for(int k = 0; k < cols_count; k++){
                    float d = read_type<float>(line,pos);
                    network.network_[i].add(j,k,d);
                }
            }

        }
        size_t rows_count = read_type<size_t>(line,pos);
        size_t cols_count = read_type<size_t>(line,pos);

        for(int j = 0; j < rows_count; j++){
            for(int k = 0; k < cols_count; k++){
                float d = read_type<float>(line,pos);
                network.output_.add(j,k,d);
            }
        }
        return network;
    }

    void train(Vector &input, Vector &expected) {
        auto result = run(input);

        Vector current_layer_errors = expected - result;


        for(int i = network_.size()-1; i >= 0; i-- ){
            auto prev_layers_error = network_[i].T() * current_layer_errors;
            Vector prev_hidden_layers_out = network_outputs_[i];
            auto deltaWeights = current_layer_errors * result;
            deltaWeights = deltaWeights * ( 1- result);
            auto deltaWeights2 = dot(deltaWeights,prev_hidden_layers_out);
            network_[i].learn(0.05,deltaWeights2);

            current_layer_errors = prev_layers_error;
            result = network_outputs_[i];

        }
        network_outputs_.clear();
    }

    auto crossover(const FeedForwardNetwork &rhs) -> FeedForwardNetwork{
        std::random_device rd;  // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        std::uniform_int_distribution<int> distrib(1, input_size_-1);
        std::uniform_int_distribution<int> distrib2(1, output_layers_-1);
        FeedForwardNetwork result(input_size_,hidden_layers_,output_layers_);
        for(std::size_t i = 0; i < hidden_layers_; ++i) {
            int rows = distrib(gen);
            int cols = distrib(gen);
            for(int j = 0; j <= rows; ++j){
                if(j == rows){
                    for(int k = 0; k <= cols; k++){
                        result.network_[i].add(j,k, network_[i].get(j,k));
                    }
                }else{
                    for(int k = 0; k < input_size_; k++){
                        result.network_[i].add(j,k, network_[i].get(j,k));
                    }
                }
            }
        }

        int rows = distrib2(gen);
        int cols = distrib(gen);
        for(int j = 0; j <= rows; ++j){
            if(j == rows){
                for(int k = 0; k <= cols; k++){
                    result.output_.add(j,k, output_.get(j,k));
                }
            }else{
                for(int k = 0; k < input_size_; k++){
                    result.output_.add(j,k, output_.get(j,k));
                }
            }
        }
        return result;
    }

    void mutate(float rate, float limit){
        std::random_device rd;  // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<float> distrib(0, 1);

        for(int i = 0; i < hidden_layers_; i++){
            for(int k = 0; k < input_size_; k++){
                for(int l = 0; l < input_size_; l++){
                    if(distrib(gen) <= rate){
                        float d = network_[i].get(k,l) + limit;
                        if (d >= 1.0){
                            d = 1.0;
                            network_[i].add(k,l,d);
                        }else if( d <=-1.0){
                            d = -1.0;
                            network_[i].add(k,l,d);
                        }else{
                            network_[i].add(k,l,d);
                        }
                    }
                }
            }
        }
        for(int k = 0; k < output_layers_; k++){
            for(int l = 0; l < input_size_; l++){
                if(distrib(gen) <= rate){
                    float d = output_.get(k,l) + limit;
                    if (d >= 1.0){
                        d = 1.0;
                        output_.add(k,l,d);
                    }else if( d <=-1.0){
                        d = -1.0;
                        output_.add(k,l,d);
                    }else{
                        output_.add(k,l,d);
                    }
                }
            }
        }
    }

    void print(){
        for(int i = 0; i < hidden_layers_+1; i++){
            std::cout<<"======================Layer "<<i<<"\n";
            network_[i].print();
        }
    }

private:
    std::size_t input_size_;
    std::size_t hidden_layers_;
    std::size_t output_layers_;

    std::vector<Matrix> network_;
    std::vector<Vector> network_outputs_;
    Matrix output_;
    cl_kernel kernel;
};

#endif //KNN_FEEDFORWARDNETWORK_H
