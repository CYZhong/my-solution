/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY 
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND 
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include <iostream>
#include <boost/timer.hpp>
#include <boost/progress.hpp>

#include "tiny_cnn.h"

#include <opencv2/opencv.hpp>
#include<strstream>

using namespace std;
using namespace cv;
//#define NOMINMAX
//#include "imdebug.h"

void sample1_3layerNN();

using namespace tiny_cnn;

//**********************************************************************************//
//定义全局变量
const int  TRNUM = 20;   //训练样本的个数
const int  TENUM = 10;    //测试样本的个数
const int  C = 3;      //分类的个数
 //**********************************************************************************//

 //**********************************************************************************//
 // convert image to vec_t
void convert_image(const string& imagefilename, double scale, int w, int h, std::vector<vec_t>& data)
{
	auto img = imread(imagefilename, IMREAD_GRAYSCALE);
	if (img.data == nullptr) return; // cannot open, or it's not an image
									 //imshow("img", img);
									 //cvWaitKey(0);
	cv::Mat_<uint8_t> resized;
	cv::resize(img, resized, cv::Size(w, h));
	vec_t d;

	std::transform(resized.begin(), resized.end(), std::back_inserter(d),
		[=](uint8_t c) { return c * scale; });
	data.push_back(d);
}

//int 转换string
string int2string(int&i) {
	strstream ss;  string str;
	ss << i; ss >> str;
	return str;
}


int main(void) {
    // construct LeNet-5 architecture
    typedef network<mse, gradient_descent_levenberg_marquardt> CNN;
    CNN nn;
    convolutional_layer<CNN, tanh_activation> C1(32, 32, 5, 1, 6);
    average_pooling_layer<CNN, tanh_activation> S2(28, 28, 6, 2);
    // connection table [Y.Lecun, 1998 Table.1]
#define O true
#define X false
    static const bool connection[] = {
        O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
        O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
        O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
        X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
        X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
        X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
    };
#undef O
#undef X
    convolutional_layer<CNN, tanh_activation> C3(14, 14, 5, 6, 16, connection_table(connection, 6, 16));
    average_pooling_layer<CNN, tanh_activation> S4(10, 10, 16, 2);
    convolutional_layer<CNN, tanh_activation> C5(5, 5, 5, 16, 120);
    fully_connected_layer<CNN, tanh_activation> F6(120, 10);

    assert(C1.param_size() == 156 && C1.connection_size() == 122304);
    assert(S2.param_size() == 12 && S2.connection_size() == 5880);
    assert(C3.param_size() == 1516 && C3.connection_size() == 151600);
    assert(S4.param_size() == 32 && S4.connection_size() == 2000);
    assert(C5.param_size() == 48120 && C5.connection_size() == 48120);

    nn.add(&C1);
    nn.add(&S2);
    nn.add(&C3);
    nn.add(&S4);
    nn.add(&C5);
    nn.add(&F6);

    std::cout << "load models..." << std::endl;

    // load MNIST dataset
    std::vector<label_t> train_labels, test_labels;
    std::vector<vec_t> train_images, test_images;


	for (int i = 1; i <= TRNUM; i++) {
		string  str0 = int2string(i);
		string trainpath = "D:\\Downloads\\train\\" + str0 + ".jpg"; //训练集路径																
		convert_image(trainpath, 1.0, 32, 32, train_images);  //转换图像格式
	}
	std::cout << "cout << train_images.size():" << train_images.size() << endl;
	//加载训练数据
	for (int i = 1; i <= TENUM; i++) {
		string  str1 = int2string(i);
		string testpath = "D:\\Downloads\\test\\" + str1 + ".jpg";  //测试集路径
		convert_image(testpath, 1.0, 32, 32, test_images);
	}
	std::cout << "cout << test_images.size():" << test_images.size() << endl;
	//**********************************************************************************//
	//【第三步】加载标签 手工添加 标签从0开始
	std::cout << "加载训练集标签" << endl;
	int Ci; //每一类训练样本的个数
	for (size_t k = 0; k < C; k++) {  // C 分类的个数
		std::cout << "第i类样本的个数：";
		std::cin >> Ci;
		for (size_t j = 1; j <= Ci; j++) {
			train_labels.push_back((label_t)k);
		}
	}
	std::cout << "train_labels" << train_labels.size() << endl;
	std::cout << "/加载测试集标签" << endl;
	int Ti; //每一类测试样本的个数
	for (size_t k = 0; k < C; k++) {  // C 分类的个数
		std::cout << "第i类样本的个数：";
		std::cin >> Ti;
		for (size_t j = 1; j <= Ti; j++) {
			test_labels.push_back((label_t)k);
		}
	}
	std::cout << "test_labels" << test_labels.size() << endl;

    std::cout << "start learning" << std::endl;

    boost::progress_display disp(train_images.size());
    boost::timer t;
    int minibatch_size = 10;

    nn.optimizer().alpha *= std::sqrt(minibatch_size);

    // create callback
    auto on_enumerate_epoch = [&](){
        std::cout << t.elapsed() << "s elapsed." << std::endl;

        tiny_cnn::result res = nn.test(test_images, test_labels);

        std::cout << nn.optimizer().alpha << "," << res.num_success << "/" << res.num_total << std::endl;

        nn.optimizer().alpha *= 0.85; // decay learning rate
        nn.optimizer().alpha = std::max(0.00001, nn.optimizer().alpha);

        disp.restart(train_images.size());
        t.restart();
    };

    auto on_enumerate_minibatch = [&](){ 
        disp += minibatch_size; 
    
        // weight visualization in imdebug
        /*static int n = 0;    
        n+=minibatch_size;
        if (n >= 1000) {
            image img;
            C3.weight_to_image(img);
            imdebug("lum b=8 w=%d h=%d %p", img.width(), img.height(), &img.data()[0]);
            n = 0;
        }*/
    };
    
    // training
    nn.train(train_images, train_labels, minibatch_size, 20, on_enumerate_minibatch, on_enumerate_epoch);

    std::cout << "end training." << std::endl;

    // test and show results
    nn.test(test_images, test_labels).print_detail(std::cout);

    // save networks
    std::ofstream ofs("LeNet-weights");
    ofs << C1 << S2 << C3 << S4 << C5 << F6;
	system("pause");
}

// learning 3-Layer Networks
void sample1_3layerNN()
{
    const int num_hidden_units = 500;
    typedef network<mse, gradient_descent> neuralnet;
    neuralnet nn;
    fully_connected_layer<neuralnet, tanh_activation> L1(28*28, num_hidden_units);
    fully_connected_layer<neuralnet, tanh_activation> L2(num_hidden_units, 10);

    // load MNIST dataset
    std::vector<label_t> train_labels, test_labels;
    std::vector<vec_t> train_images, test_images;

    parse_mnist_labels("train-labels.idx1-ubyte", &train_labels);
    parse_mnist_images("train-images.idx3-ubyte", &train_images, -1.0, 1.0, 0, 0);
    parse_mnist_labels("t10k-labels.idx1-ubyte", &test_labels);
    parse_mnist_images("t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 0, 0);

    nn.add(&L1);
    nn.add(&L2);
    nn.optimizer().alpha = 0.001;
    
    boost::progress_display disp(train_images.size());
    boost::timer t;

    // create callback
    auto on_enumerate_epoch = [&](){
        std::cout << t.elapsed() << "s elapsed." << std::endl;

        tiny_cnn::result res = nn.test(test_images, test_labels);

        std::cout << nn.optimizer().alpha << "," << res.num_success << "/" << res.num_total << std::endl;

        nn.optimizer().alpha *= 0.85; // decay learning rate
        nn.optimizer().alpha = std::max(0.00001, nn.optimizer().alpha);

        disp.restart(train_images.size());
        t.restart();
    };

    auto on_enumerate_data = [&](){ 
        ++disp; 
    };  

    nn.train(train_images, train_labels, 1, 20, on_enumerate_data, on_enumerate_epoch);
}