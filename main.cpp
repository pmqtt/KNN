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
    std::string value = "";
    for(int i = 0; i < count; i++){
        if((int)(v.get(i)) == 1 ){
            value += std::to_string(i);
        }
    }
    if(value == ""){
        return 11;
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

float calculate_fintess(FeedForwardNetwork &f,uchar** images,uchar* labels){
    float fitness = 0.0;
    for (int i = 0; i < 100; ++i) {
        auto image = images[i];
        Vector v(784);
        for (std::size_t i = 0; i < 784; ++i) {
            float d = (float) (image[i]) / 255;
            v.add(i, d);
        }
        v = f.run(v);
        v.round();
        fitness += std::abs((int)fitness_value(v, 10) - (int) (labels[i]));
    }
    return fitness;
}

void print_pop_fitness(const std::vector<float> & pop){
    int i = 1;
    for(const auto & x: pop){
        std::cout<<i<<": "<< x <<" ";
        i++;
    }
    std::cout<<"\n";
}

std::vector<float> calculate_population_fitness(std::vector<FeedForwardNetwork> &networks,uchar** images,uchar* labels){
    std::vector<float> population_fitness;
    for(auto & f: networks){
        population_fitness.push_back(calculate_fintess(f,images,labels));
    }
    return population_fitness;
}

std::vector<FeedForwardNetwork> sex(std::vector<FeedForwardNetwork> & networks){
    std::vector<FeedForwardNetwork> childs;
    for(int i = 0; i < networks.size(); ++i){
        for(int j = 0; j <networks.size(); j++){
            if(i != j) {
                auto child = networks[i].crossover(networks[j]);
                childs.push_back(child);
            }
        }
    }
    return childs;
}
void mutate(std::vector<FeedForwardNetwork> & networks){
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<float> distrib(-0.003, 0.003);
    for(auto &x: networks){
        x.mutate(0.1,distrib(gen));
    }
}

Vector create_from_label(int label){
    Vector res(10);
    for(int i = 0; i < 10; ++i){
        if(label == i){
            res.add(i,0.9);
        }else {
            res.add(i, 0.1);
        }
    }
    return res;
}
#include "gpu.h"
#if 1
int main(int argc, char **argv){
    int count;
    int cnt;
    int sz;
    auto labels = read_mnist_labels("label.txt",count);
    auto images = read_mnist_images("images.txt",cnt,sz);

    FeedForwardNetwork network(784,2,10);


    for(int i = 0; i < 10000; ++i) {
        auto image = images[i];
        Vector v(784);
        for (std::size_t j = 0; j < 784; ++j) {
            float d = (float) (image[j]) / 255;
            v.add(j, d);
        }
        Vector expected = create_from_label((int)labels[i]);
        network.train(v,expected);
        std::cout<<"Trained: "<<i<<"\n";
    }
    int goals = 0;
    for(int i = 10000; i < 11000; ++i) {
        auto image = images[i];
        Vector v(784);
        for (std::size_t j = 0; j < 784; ++j) {
            float d = (float) (image[j]) / 255;
            v.add(j, d);
        }
        auto res = network.run(v);
        //res.round();
        std::cout<<"LABEL: "<<(int)labels[i]<<": ";
        res.round();
        res.print();
        std::cout<< " CALCULATED: "<< fitness_value(res,10);
        if((int)labels[i] == fitness_value(res,10)){
            goals++;
        }
        std::cout<<"\n";
    }
    std::cout<<"Correct FOUNDS: "<<goals<<"\n";

    //network.print();

    delete [] labels;
    for(int i = 0; i < count; ++i){
        delete [] images[i];
    }
    delete[] images;


#if 0
    float fitness = 0.0;
    float crank =  100000000000000000;



    std::vector<std::string> files ={
            "population/70.net","population/167.net",
            "population/367.net","population/464.net",
            "population/3146.net","population/4164"
    };
    std::vector<FeedForwardNetwork> networks;

    for(const auto &file: files){
        networks.push_back(FeedForwardNetwork::load(file));
    }

    auto population_fitness = calculate_population_fitness(networks,images,labels);
    print_pop_fitness(population_fitness);

    auto childs = sex(networks);
    mutate(childs);
    population_fitness = calculate_population_fitness(childs,images,labels);
    print_pop_fitness(population_fitness);







#endif


#if 0
    FeedForwardNetwork f = FeedForwardNetwork::load("population/70.net");
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<float> distrib(-0.005, 0.005);
    int max_founds = -1;
    float mutation = 0.5;
    for(int k = 0; k < 100000; k++) {
        int founds = 0;
        for (int i = 0; i < 100; ++i) {
            auto image = images[i];
            Vector v(784);
            for (std::size_t i = 0; i < 784; ++i) {
                float d = (float) (image[i]) / 255;
                v.add(i, d);
            }
            v = f.run(v);
            v.round();
            if (fitness_value(v, 10) < 10) {
                founds++;
            }
        }
        if(founds > max_founds) {
            f.store("Found.net");
            if(founds == 100) {
                return 0;
            }
            max_founds = founds;
            mutation -= 0.002;

        }else if(founds < max_founds){
            mutation += 0.002;
        }
        if(mutation > 1.0){
            mutation = 1.0;
        }
        std::cout<<"k: " << k<< "found:"<<founds<< " mutation:"<< mutation<<"\n";

        f.mutate(mutation,distrib(gen));
    }
#endif
#if 0
    std::cout<<"Count:"<<count<<"\n";
    std::cout<<"SZ:"<<sz<<"\n";
    std::vector<FeedForwardNetwork> population;
    int population_count = 0;
    for(int k = 0; k < count; k++) {

        FeedForwardNetwork net(784,2,10);
        Vector v(784);
        auto image = images[k];
        for (std::size_t i = 0; i < 784; ++i) {
            float d = (float) (image[i]) / 255;
            v.add(i, d);
        }
        v = net.run(v);
        v.round();
        if(fitness_value(v, 10) < 10 ){
            population_count++;
            population.push_back(net);
            std::cout<<"PopulationCount:"<<population_count<<"\n";
            net.store(std::to_string(k) +".net");
            v.print();
        }
        std::cout<<"k:"<<k<<"\n";
        //std::cout << "\n" << (int) labels[0] << "\n";
    }
    std::cout<<"Population Size:"<<population.size()<<"\n";
#endif
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




















