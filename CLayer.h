#pragma once
#include "Imagelib.h"
#include "CTensor.h"
#include <omp.h> // OpenMP ��� ����

#define MEAN_INIT 0
#define LOAD_INIT 1

// Layer�� tensor�� ��/������� ������, Ư�� operation�� �����ϴ� Convolutional Neural Netowork�� �⺻ ���� ����

class Layer {
protected:
	int fK; // kernel size in K*K kernel
	int fC_in; // number of channels
	int fC_out; //number of filters
	string name;
public:
	Layer(string _name, int _fK, int _fC_in, int _fC_out) : name(_name), fK(_fK), fC_in(_fC_in), fC_out(_fC_out) {}
	virtual ~Layer() {}; //����Ҹ��� (����: https://wonjayk.tistory.com/243)
	virtual Tensor3D* forward(const Tensor3D* input) = 0;
	//	virtual bool backward() = 0;
	virtual void print() const = 0;
	virtual void get_info(string& _name, int& _fK, int& _fC_in, int& _fC_out) const = 0;
};


class Layer_ReLU : public Layer {
public:
	Layer_ReLU(string _name, int _fK, int _fC_in, int _fC_out) :Layer(_name, _fK, _fC_in, _fC_out)
	{
		// (������ ��)
		// ����1: Base class�� �����ڸ� ȣ���Ͽ� �ɹ� ������ �ʱ�ȭ �� ��(�ݵ�� initialization list�� ����� ��)
	}
	~Layer_ReLU() {}
	Tensor3D* forward(const Tensor3D* input) override {
		// (������ ��)
		int nH;
		int nW;
		int nC;
		input->get_info(nH, nW, nC);
		Tensor3D* output = new Tensor3D(nH, nW, nC);

		// OpenMP: �� ä��(ch)�� ���� ������ ���� �������̹Ƿ� ���� ó��
#pragma omp parallel for
		for (int ch = 0; ch < nC; ch++) {
			for (int y = 0; y < nH; y++) {
				for (int x = 0; x < nW; x++) {
					double val = input->get_elem(y, x, ch);
					output->set_elem(y, x, ch, val > 0 ? val : 0);
				}
			}
		}
		// ����1: input tensor�� ���� �� element x�� ����̸� �״�� ����, �����̸� 0���� output tensor�� �����Ұ�    
		// ����2: �̶�, output tensor�� �����Ҵ��Ͽ� �ּҰ��� ��ȯ�� ��
		// �Լ�1: Tensor3D�� �ɹ��Լ��� get_info(), get_elem(), set_elem()�� ������ Ȱ���� ��

		cout << name << " is finished" << endl;
		return output;
	};
	void get_info(string& _name, int& _fK, int& _fC_in, int& _fC_out) const override {
		// (������ ��)
		_name = name;
		_fK = fK;
		_fC_in = fC_in;
		_fC_out = fC_out;
		// ����: Tensor3D�� get_info()�� ���������� �ɹ� �������� pass by reference�� �ܺο� ����
	}
	void print() const override {
		// (������ ��)
		cout << name << ":\t" << fK << " * " << fK << " * " << fC_in << " * " << fC_out << endl;
		// ����: Tensor3D�� print()�� ���������� ������ ũ�⸦ ȭ�鿡 ���
	}
};



class Layer_Conv : public Layer {
private:
	string filename_weight;
	string filename_bias;
	double**** weight_tensor; // fK x fK x _fC_in x _fC_out ũ�⸦ ������ 4���� �迭
	double* bias_tensor;     // _fC_out ũ�⸦ ������ 1���� �迭 (bias�� �� filter�� 1�� ����) 
public:
	Layer_Conv(string _name, int _fK, int _fC_in, int _fC_out, int init_type, string _filename_weight = "", string _filename_bias = "") :
		Layer(_name, _fK, _fC_in, _fC_out), filename_weight(_filename_weight), filename_bias(_filename_bias)
	{
		// (������ ��)
		// ����1: initialization list�� base class�� �����ڸ� �̿��Ͽ� �ɹ� ������ �ʱ�ȭ �� ��
		// ����2: filename_weight�� filename_bias�� LOAD_INIT ����� ��� �ش� ���Ϸκ��� ����ġ/���̾�� �ҷ���
		// ����3: init() �Լ��� init_type�� �Է����� �޾� ����ġ�� �ʱ�ȭ �� 
		// �Լ�1: dmatrix4D()�� dmatrix1D()�� ����Ͽ� 1����, 4���� �迭�� ���� �Ҵ��� ��
		init(init_type);
	}
	void init(int init_type) {
		// (������ ��)
		// ����1: init_type (MEAN_INIT �Ǵ� LOAD_INIT)�� ���� ����ġ�� �ٸ� ������� �ʱ�ȭ ��
		// ����2: MEAN_INIT�� ��� ���ʹ� ��հ��� �����ϴ� ���Ͱ� �� (��, ��� ����ġ ���� ������ ũ��(fK*fK*fC_in)�� ������ ������ (�̶� bias�� ��� 0���� ����)
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
		// ����3: LOAD_INIT�� ��� filename_weight, filename_bias�� �̸��� ������ ������ ���� �о� ����ġ�� ����(�ʱ�ȭ) ��  
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
		// �Լ�1: dmatrix4D()�� dmatrix1D()�� ����Ͽ� 1����, 4���� �迭�� ���� �Ҵ��� ��
	}
	~Layer_Conv() override {
		// (������ ��)
		// ����1: weight_tensor�� bias_tensor�� ���� �Ҵ� ������ ��
		free_dmatrix4D(weight_tensor, fK, fK, fC_in, fC_out);
		free_dmatrix1D(bias_tensor, fC_out);
		// �Լ�1: free_dmatrix4D(), free_dmatrix1D() �Լ��� ���
	}
	Tensor3D* forward(const Tensor3D* input) override {
		// (������ ��)
		int nH, nW, nC;
		input->get_info(nH, nW, nC);
		Tensor3D* output = new Tensor3D(nH, nW, fC_out);
		int offset = (fK - 1) / 2;

		// OpenMP: �� ��� ä��(out)�� ���� ������� ������ ���� �������̹Ƿ� ���� ó��
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
					// Bias�� ������� �հ谡 ���� �� �� ���� ����
					output->set_elem(y, x, out, sum + bias_tensor[out]);
				}
			}
		}

		cout << name << " is finished" << endl;
		return output;
	};

	void get_info(string& _name, int& _fK, int& _fC_in, int& _fC_out) const override {
		// (������ ��)
		_name = name;
		_fK = fK;
		_fC_in = fC_in;
		_fC_out = fC_out;
		// ����: Layer_ReLU�� ����
	}
	void print() const override {
		// (������ ��)
		cout << name << ":\t" << fK << " * " << fK << " * " << fC_in << " * " << fC_out << endl;
		// ����: Layer_ReLU�� ����
	}
};