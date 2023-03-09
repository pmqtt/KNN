#include "FeedForwardNetwork.h"
#include <iostream>
#include <string>
#include <fstream>
#include <stdexcept>

typedef unsigned char uchar;

auto create_population(std::size_t count) -> std::vector<FeedForwardNetwork>{
    std::vector<FeedForwardNetwork> result;
    for(std::size_t i = 0; i < count; ++i){
        FeedForwardNetwork net(784,2,10);
        result.push_back(net);
    }
    return result;
}


auto fitness_value(const Vector & v, std::size_t count) -> std::size_t{
    std::string value;
    int k = 9;
    for(int i = 0; i < count; i++){
        if((int)(v.get(i)) == 1 ){
            value += std::to_string(k);
        }
        k--;
    }
    return atoi(value.c_str());
}


uchar* read_mnist_labels(const std::string & full_path, int& number_of_labels) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };


    std::ifstream file(full_path, std::ios::binary);

    if(file.is_open()) {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2049) throw std::runtime_error("Invalid MNIST label file!");

        file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        uchar* _dataset = new uchar[number_of_labels];
        for(int i = 0; i < number_of_labels; i++) {
            file.read((char*)&_dataset[i], 1);
        }
        return _dataset;
    } else {
        throw std::runtime_error("Unable to open file `" + full_path + "`!");
    }
}

uchar** read_mnist_images(const std::string full_path, int& number_of_images, int& image_size) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };


    std::ifstream file(full_path, std::ios::binary);

    if(file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051) throw std::runtime_error("Invalid MNIST image file!");

        file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;

        uchar** _dataset = new uchar*[number_of_images];
        for(int i = 0; i < number_of_images; i++) {
            _dataset[i] = new uchar[image_size];
            file.read((char *)_dataset[i], image_size);
        }
        return _dataset;
    } else {
        throw std::runtime_error("Cannot open file `" + full_path + "`!");
    }
}

#if 1
int main(int argc, char **argv){
    int count;
    int cnt;
    int sz;
    auto labels = read_mnist_labels("label.txt",count);

    auto images = read_mnist_images("images.txt",cnt,sz);

    std::cout<<"Count:"<<count<<"\n";
    std::cout<<"SZ:"<<sz<<"\n";
    std::vector<FeedForwardNetwork> population;
    int population_count = 0;
    for(int k = 0; k < count; k++) {
        Vector v(784);
        FeedForwardNetwork net(784,2,10);
        auto image = images[k];
        for (std::size_t i = 0; i < 784; ++i) {
            double d = (double) (image[i]) / 255;
            v.add(i, d);
        }
        v = net.run(v);
        v.round();
        if(fitness_value(v, 10) < 10 ){
            population_count++;
            population.push_back(net);
            std::cout<<"PopulationCount:"<<population_count<<"\n";
        }
        std::cout<<"k:"<<k<<"\n";
        //std::cout << "\n" << (int) labels[0] << "\n";
    }
    std::cout<<"Population Size:"<<population.size()<<"\n";
}

#endif

#if 0
int main(int argc,char ** argv){
    Matrix m(5,8);
    Vector v(8);
    for(std::size_t i = 0; i < 8; ++i){
        v.add(i,1.0);
    }
    m.fill_random();
    m.print();
    std::cout<<"\n\t * \n ";
    v.print();
    
    std::cout<<"\n\t = \n ";
    auto x = m * v;
    x.print();
    return 0;
}

#endif




















