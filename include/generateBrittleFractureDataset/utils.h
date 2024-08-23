#ifndef UTILS_H
#define UTILS_H

#include "materials.h"
#include "omp.h"
#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <set>
#include <iomanip>
#include <map>
#include <string.h>
#include <assert.h> 
#include <random>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>


#include <sstream>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <cfloat>

const double PI = 3.141592653589793238463L;

// Used to store elements which are w, w1, w11, w12, w2, w21, w22
// 1) w is the weight. 2) w1 is the weight wrt the first axis. 3) w11 is the second-order derivative wrt the first axis.
// 4) w12 is the compound partial derivative which is the same as w21.
// 5) w2 is the weight wrt the second axis. 6) It is equal to w12. 7) w22 is the second-order derivative wrt the second axis.
typedef Eigen::Matrix<double, 7, 1> Vector7d;
typedef std::complex<double> DComplex;

struct parametersSim {

	// number of parallel threads
	int numOfThreads = 6;

	Eigen::Vector3d gravity = {0 , 0 , 0 };


	Eigen::Vector3d force_position = {0 , 0 , 0 };
	Eigen::Vector3d force_direction = {0 , 0 , 0 };
	double force_magnitude = 0;

	// Bcakground Eulerian grid
	double dx = 2.0E-3; // grid space
	double DP = dx * dx / 4.0;

	// material properties
	double dt = 1.0E-6; // timestep


	// columb friction
	double kc = 1.0;
	double gama = 0.8;
	double cF = 0.2; // columb friction coefficient

	// damping coefficient
	double nu = 0;

	double damageThreshold = 0.97;



	void updateDenpendecies()
	{
		DP = dx * dx / 4;
	}


};



// Struct of obj mesh
struct objMesh
{
	std::vector<Eigen::Vector3d> vertices;
	std::vector<Eigen::Vector3i> faces;

	std::pair<Eigen::Vector3d, Eigen::Vector3d> minMaxCoor;

	void calMinMaxCoor();
	void resizeAndRemove(double size); // resize and remove to region ((1-s)/2,(1-s)/2,(1-s)/2) -> ((1+s)/2,(1+s)/2,(1+s)/2)

};


struct meshObjFormat {
	// input mesh vertices and faces
	std::vector<Eigen::Vector3d> vertices;
	std::vector<std::vector<int>> faces;
	std::vector<int> faceFromVoroCell; // indicate the voronoi cell which the face belongs to
	std::vector<int> faceFromtheOtherVoroCell; // indicate the voronoi cell which the face belongs to(the other side)
};


int generateRandomInt(int min, int max);

double generateRandomDouble(double min, double max, std::random_device rd);

int calculateID(int x, int y, int z, Eigen::Vector3d len, double dx); // coordinate of x and y, length in three dimensions of the cube, grid space

std::string calculateID_string(int x, int y, int z); // string id of the particle

// split a line from a text file
std::vector<std::string> split(const std::string& s, const std::string& seperator);

// Given the vertices and faces information, write the obj file.
void writeObjFile(std::vector<Eigen::Vector3d> vertices, std::vector<std::vector<int>> faces, std::string name, bool startFrom0);

// Given the vertices information, write an obj file and an txt file.
void writeObjFile(std::vector<Eigen::Vector3d> vertices, std::string name);

double calculateTriangleArea3D(Eigen::Vector3d p0, Eigen::Vector3d p1, Eigen::Vector3d p2);

// give a Eigen::Vector3d, calculate its skew matrix
Eigen::Matrix3d getSkewMatrix(Eigen::Vector3d vc);

// partial derivative of matrix multiplication wrt matrix
// Let C=AB, where C, A, B are 3x3 matrices. This function calculate the derivative of \frac{\partial{C}}{\partial{A}}
// and the result is a 3x3 matrix where each element is \frac{\partial{C_{ij}}}{\partial{A}} whihc is also a 3x3 matrix
std::map<int, std::map<int, Eigen::Matrix3d>> calMatrixProductDerivative(Eigen::Matrix3d& B);

#endif