// g2o0.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>

using namespace std;

class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		virtual void setToOriginImpl() 
	{
		_estimate << 0, 0, 0;
	}

	virtual void oplusImpl(const double* update) 
	{
		_estimate += Eigen::Vector3d(update);
	}

	virtual bool read(istream& in) { return true; }
	virtual bool write(ostream& out) const { return true; }
};

// 误差模型 模板参数：观测值维度，类型，连接顶点类型
class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}
	// 计算曲线模型误差
	void computeError()
	{
		const CurveFittingVertex* v = static_cast<const CurveFittingVertex*> (_vertices[0]);
		const Eigen::Vector3d abc = v->estimate();
		_error(0, 0) = _measurement - std::exp(abc(0, 0)*_x*_x + abc(1, 0)*_x + abc(2, 0));
	}
	virtual bool read(istream& in) { return true; }
	virtual bool write(ostream& out) const { return true; }
public:
	double _x;  // x 值， y 值为 _measurement
};

class CurveFittingEdge_1 : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		CurveFittingEdge_1(double x, double y) : BaseUnaryEdge(), _x(x), _y(y) {}
	// 计算曲线模型误差
	void computeError()
	{
		const CurveFittingVertex* v = static_cast<const CurveFittingVertex*> (_vertices[0]);
		const Eigen::Vector3d abc = v->estimate();
		_error(0, 0) = _y - std::exp(abc(0, 0)*_x*_x + abc(1, 0)*_x + abc(2, 0));
	}
	virtual bool read(istream& in) { return true; }
	virtual bool write(ostream& out) const { return true; }
public:
	double _x;  // x 值， 
	double _y;	//y 值
};

int main()
{
	double a = 1.0, b = 2.0, c = 1.0;     // 真实参数值
	int N = 100;                          // 数据点
	double w_sigma = 0.3;                 // 噪声Sigma值
	cv::RNG rng;                          // 随机数产生器
	double abc[3] = { 0,0,0 };            // abc参数的估计值

	vector<double> x_data, y_data;        // 数据

	cout << "generating data: " << endl;
	for (int i = 0; i<N; i++)
	{
		double x = i / 100.0;
		x_data.push_back(x);
		double noise = rng.gaussian(w_sigma);
		if (i % 10 == 0) noise *= 10;
		y_data.push_back(
			exp(a*x*x + b * x + c) + noise
		);
		cout << x_data[i] << " " << y_data[i] <<" "<<noise << endl;
	}

	// 构建图优化，先设定g2o
	typedef g2o::BlockSolver< g2o::BlockSolverTraits<3, 1> > MyBlock;  // 每个误差项优化变量维度为3，误差值维度为1
	typedef g2o::LinearSolverDense<MyBlock::PoseMatrixType> MyLinear;
//	std::unique_ptr<MyBlock::LinearSolverType> linearSolver = g2o::make_unique<MyLinear>(); // 线性方程求解器
//	std::unique_ptr<MyBlock> solver_ptr = g2o::make_unique<MyBlock>( g2o::make_unique<MyLinear>() );      // 矩阵块求解器
	//MyBlock* solver_ptr = g2o::make_unique<MyBlock>(g2o::make_unique<MyLinear>());
	// 梯度下降方法，从GN, LM, DogLeg 中选
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
		g2o::make_unique<MyBlock>(g2o::make_unique<MyLinear>()));


	g2o::SparseOptimizer optimizer;     // 图模型
	optimizer.setAlgorithm(solver);   // 设置求解器
	optimizer.setVerbose(true);       // 打开调试输出

	// 往图中增加顶点
	CurveFittingVertex* v = new CurveFittingVertex();
	v->setEstimate(Eigen::Vector3d(0, 0, 0));
	v->setId(0);
	optimizer.addVertex(v);

	// 往图中增加边
	for (int i = 0; i<N; i++)
	{
		CurveFittingEdge_1* edge = new CurveFittingEdge_1(x_data[i], y_data[i]);
		edge->setId(i);
		edge->setVertex(0, v);                // 设置连接的顶点
		edge->setMeasurement(y_data[i]);      // 观测数值
		edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma*w_sigma)); // 信息矩阵：协方差矩阵之逆
		g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
		edge->setRobustKernel(rk);
		optimizer.addEdge(edge);
	}

	// 执行优化
	cout << "start optimization" << endl;
	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
	optimizer.initializeOptimization();
	optimizer.optimize(100);
	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
	cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

	// 输出优化值
	Eigen::Vector3d abc_estimate = v->estimate();
	cout << "estimated model: " << abc_estimate.transpose() << endl;
	int i = 0;
	for (auto it : optimizer.edges()) {
		CurveFittingEdge_1* edge = dynamic_cast<CurveFittingEdge_1*>(it);
		cout << edge->id() << " " <<edge->error()[0] << " "<< edge->chi2() << endl;
	}
	
    return 0;
}

