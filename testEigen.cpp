#include<iostream>
#include<Eigen\Core>
#include<Eigen\Dense>
#include <random>
using namespace Eigen;

int testEigen()
{
	//validate 
	//The Quaterniond is from sphere.g2o, note the order.
//	Eigen::Quaterniond q0 = Eigen::Quaterniond(-4.3325e-17, 0.706662, 4.32706e-17, 0.707551);
//	Eigen::Quaterniond q1 = Eigen::Quaterniond(0.0167761, 0.690942, -0.0699593, 0.719321);// -0.129409, -0.224144, -0.482965, 0.836515);// -0.127577, -0.224431, -0.491096, 0.831973);
//	Eigen::Quaterniond q01 = Eigen::Quaterniond(0.997479 , -0.0634389, 0.0313134, 0.00554102);// 0.61035, 0

	Eigen::Quaterniond q0 = Eigen::Quaterniond(-5.30288e-17, 0.5, 3.06162e-17, 0.866025);
	Eigen::Quaterniond q1 = Eigen::Quaterniond(0.836515, -0.129409, -0.224144, -0.482965);
	Eigen::Quaterniond q01 = Eigen::Quaterniond(0.482964, 0.612372, 0.129411, 0.612372);
	Eigen::Matrix3d r0(q0);
	Eigen::Matrix3d r1(q1);
	Eigen::Matrix3d r01(q01);
	Eigen::Quaterniond q01_ = q0.inverse()*q1;
	Eigen::Matrix3d r01_ = r0.inverse()*r1;
	std::cout << "q0: " << std::endl;
	std::cout << (q0).matrix() << std::endl;
	std::cout << "q01: " << std::endl;
	std::cout << q01.matrix() << std::endl;
	std::cout << "q1: " << std::endl;
	std::cout << q1.matrix() << std::endl << std::endl;

	std::cout << (q01_.inverse()*q01).matrix() << std::endl;
	for(int i = 0; i < 4; ++i){
		std::cout << q01.coeffs()[i] << " ";
	}
	std::cout << std::endl;
	for (int i = 0; i < 4; ++i) {
		std::cout << q01_.coeffs()[i] << " " ;
	}
	std::cout << std::endl;
	return 0;
}

int main()
{
	//from g2o project create_sphere
	//VertexSE3* prev = vertices[i-1];
    //VertexSE3* cur  = vertices[i];
    //Eigen::Isometry3d t = prev->estimate().inverse() * cur->estimate();
	//means  T_01 = T_0.inv * T_1; T_1 = T_0*T_01
	testEigen();
	return 0;
}