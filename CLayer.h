#pragma once
#include "Imagelib.h"
#include "CTensor.h"
#include <omp.h> // OpenMP 헤더 포함

#define MEAN_INIT 0
#define LOAD_INIT 1

// Layer는 tensor를 입/출력으로 가지며, 특정 operation을 수행하는 Convolutional Neural Netowork의 기본 연산 단위

class Layer {
protected:
	int fK; // kernel size in K*K kernel
	int fC_in; // number of channels
	int fC_out; //number of filters
	string name;
public:
	Layer(string _name, int _fK, int _fC_in, int _fC_out) : name(_name), fK(_fK), fC_in(_fC_in), fC_out(_fC_out) {}
	virtual ~Layer() {}; //가상소멸자 (참고: https://wonjayk.tistory.com/243)
	virtual Tensor3D* forward(const Tensor3D* input) = 0;
	//	virtual bool backward() = 0;
	virtual void print() const = 0;
	virtual void get_info(string& _name, int& _fK, int& _fC_in, int& _fC_out) const = 0;
};


class Layer_ReLU : public Layer {
public:
	Layer_ReLU(string _name, int _fK, int _fC_in, int _fC_out) :Layer(_name, _fK, _fC_in, _fC_out)
	{
		// (구현할 것)
		// 동작1: Base class의 생성자를 호출하여 맴버 변수를 초기화 할 것(반드시 initialization list를 사용할 것)
	}
	~Layer_ReLU() {}
	Tensor3D* forward(const Tensor3D* input) override {
		// (구현할 것)
		int nH;
		int nW;
		int nC;
		input->get_info(nH, nW, nC);
		Tensor3D* output = new Tensor3D(nH, nW, nC);

		// OpenMP: 각 채널(ch)에 대한 연산은 서로 독립적이므로 병렬 처리
#pragma omp parallel for
		for (int ch = 0; ch < nC; ch++) {
			for (int y = 0; y < nH; y++) {
				for (int x = 0; x < nW; x++) {
					double val = input->get_elem(y, x, ch);
					output->set_elem(y, x, ch, val > 0 ? val : 0);
				}
			}
		}
		// 동작1: input tensor에 대해 각 element x가 양수이면 그대로 전달, 음수이면 0으로 output tensor에 전달할것    
		// 동작2: 이때, output tensor는 동적할당하여 주소값을 반환할 것
		// 함수1: Tensor3D의 맴버함수인 get_info(), get_elem(), set_elem()을 적절히 활용할 것

		cout << name << " is finished" << endl;
		return output;
	};
	void get_info(string& _name, int& _fK, int& _fC_in, int& _fC_out) const override {
		// (구현할 것)
		_name = name;
		_fK = fK;
		_fC_in = fC_in;
		_fC_out = fC_out;
		// 동작: Tensor3D의 get_info()와 마찬가지로 맴버 변수들을 pass by reference로 외부에 전달
	}
	void print() const override {
		// (구현할 것)
		cout << name << ":\t" << fK << " * " << fK << " * " << fC_in << " * " << fC_out << endl;
		// 동작: Tensor3D의 print()와 마찬가지로 차원의 크기를 화면에 출력
	}
};



class Layer_Conv : public Layer {
private:
	string filename_weight;
	string filename_bias;
	double**** weight_tensor; // fK x fK x _fC_in x _fC_out 크기를 가지는 4차원 배열
	double* bias_tensor;     // _fC_out 크기를 가지는 1차원 배열 (bias는 각 filter당 1개 존재) 
public:
	Layer_Conv(string _name, int _fK, int _fC_in, int _fC_out, int init_type, string _filename_weight = "", string _filename_bias = "") :
		Layer(_name, _fK, _fC_in, _fC_out), filename_weight(_filename_weight), filename_bias(_filename_bias)
	{
		// (구현할 것)
		// 동작1: initialization list와 base class의 생성자를 이용하여 맴버 변수를 초기화 할 것
		// 동작2: filename_weight와 filename_bias는 LOAD_INIT 모드일 경우 해당 파일로부터 가중치/바이어스를 불러옴
		// 동작3: init() 함수는 init_type를 입력으로 받아 가중치를 초기화 함 
		// 함수1: dmatrix4D()와 dmatrix1D()를 사용하여 1차원, 4차원 배열을 동적 할당할 것
		init(init_type);
	}
	void init(int init_type) {
		// (구현할 것)
		// 동작1: init_type (MEAN_INIT 또는 LOAD_INIT)에 따라 가중치를 다른 방식으로 초기화 함
		// 동작2: MEAN_INIT의 경우 필터는 평균값을 산출하는 필터가 됨 (즉, 모든 가중치 값이 필터의 크기(fK*fK*fC_in)의 역수와 같아짐 (이때 bias는 모두 0으로 설정)
		if (init_type == MEAN_INIT) {
			int vol = fK * fK * fC_in;
			weight_tensor = dmatrix4D(fK, fK, fC_in, fC_out);
			double weight = 1.0 / vol;


			for (int o = 0; o < fC_out; o++)
				for (int i = 0; i < fC_in; i++)
					for (int y = 0; y < fK; y++)
						for (int x = 0; x < fK; x++)
							weight_tensor[y][x][i][o] = weight;
			bias_tensor = dmatrix1D(fC_out);
			for (int i = 0; i < fC_out; i++)
				bias_tensor[i] = 0;
		}
		// 동작3: LOAD_INIT의 경우 filename_weight, filename_bias의 이름을 가지는 파일의 값을 읽어 가중치에 저장(초기화) 함  
		else if (init_type == LOAD_INIT) {
			double tmp;
			weight_tensor = dmatrix4D(fK, fK, fC_in, fC_out);
			bias_tensor = dmatrix1D(fC_out);

			ifstream fin_1(filename_weight);
			ifstream fin_2(filename_bias);

			for (int o = 0; o < fC_out; o++) {
				for (int i = 0; i < fC_in; i++) {
					for (int y = 0; y < fK; y++) {
						for (int x = 0; x < fK; x++) {
							fin_1 >> tmp;
							weight_tensor[y][x][i][o] = tmp;
						}
					}
				}
			}

			for (int i = 0; i < fC_out; i++) {
				fin_2 >> tmp;
				bias_tensor[i] = tmp;
			}
			fin_1.close();
			fin_2.close();
		}
		// 함수1: dmatrix4D()와 dmatrix1D()를 사용하여 1차원, 4차원 배열을 동적 할당할 것
	}
	~Layer_Conv() override {
		// (구현할 것)
		// 동작1: weight_tensor와 bias_tensor를 동적 할당 해제할 것
		free_dmatrix4D(weight_tensor, fK, fK, fC_in, fC_out);
		free_dmatrix1D(bias_tensor, fC_out);
		// 함수1: free_dmatrix4D(), free_dmatrix1D() 함수를 사용
	}
	Tensor3D* forward(const Tensor3D* input) override {
		// (구현할 것)
		int nH, nW, nC;
		input->get_info(nH, nW, nC);
		Tensor3D* output = new Tensor3D(nH, nW, fC_out);
		int offset = (fK - 1) / 2;

		// OpenMP: 각 출력 채널(out)에 대한 컨볼루션 연산은 서로 독립적이므로 병렬 처리
#pragma omp parallel for
		for (int out = 0; out < fC_out; out++) {
			for (int y = offset; y < nH - offset; y++) {
				for (int x = offset; x < nW - offset; x++) {
					double sum = 0.0;
					for (int in = 0; in < nC; in++) {
						for (int ph = 0; ph < fK; ph++) {
							for (int pw = 0; pw < fK; pw++) {
								sum += weight_tensor[ph][pw][in][out] *
									input->get_elem(y + ph - offset, x + pw - offset, in);
							}
						}
					}
					// Bias는 컨볼루션 합계가 끝난 후 한 번만 더함
					output->set_elem(y, x, out, sum + bias_tensor[out]);
				}
			}
		}

		cout << name << " is finished" << endl;
		return output;
	};

	void get_info(string& _name, int& _fK, int& _fC_in, int& _fC_out) const override {
		// (구현할 것)
		_name = name;
		_fK = fK;
		_fC_in = fC_in;
		_fC_out = fC_out;
		// 동작: Layer_ReLU와 동일
	}
	void print() const override {
		// (구현할 것)
		cout << name << ":\t" << fK << " * " << fK << " * " << fC_in << " * " << fC_out << endl;
		// 동작: Layer_ReLU와 동일
	}
};