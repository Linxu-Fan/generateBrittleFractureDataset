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
#include <Eigen/Core>
#include <Eigen/Eigenvalues>


#include <sstream>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <cfloat>



// Used to store elements which are w, w1, w11, w12, w2, w21, w22
// 1) w is the weight. 2) w1 is the weight wrt the first axis. 3) w11 is the second-order derivative wrt the first axis.
// 4) w12 is the compound partial derivative which is the same as w21.
// 5) w2 is the weight wrt the second axis. 6) It is equal to w12. 7) w22 is the second-order derivative wrt the second axis.
typedef Eigen::Matrix<double, 7, 1> Vector7d;

struct parametersSim {

	// number of parallel threads
	int numOfThreads = 6;

	Eigen::Vector3d gravity = {0 , 0 , 0 };

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



	void updateDenpendecies()
	{
		DP = dx * dx / 4;
	}


};


struct Mesh
{
	/////////////////////////////////////////////
	// information about whole mesh
	/////////////////////////////////////////////
	double thickness = -99;


	/////////////////////////////////////////////
	// information about mesh nodes
	/////////////////////////////////////////////
	std::vector<int> corresMPMPar_node; // the corresponding MPM particle of this node
	std::vector<Eigen::Vector3d> pos_node; // position of each node
	std::vector<Eigen::Vector3d> vel_node; // velocity of each node
	std::vector<double> vol_node; // volume of each node
	std::vector<bool> ifBoundary_node; // if this node is in the boundary or not
	std::vector<bool> boundary_node; // if the node is in the boundary or not



	/////////////////////////////////////////////
	// information about mesh edges
	/////////////////////////////////////////////
	std::map<std::string, Eigen::Vector2i> edgeTris; // triangle(s) that share this edge. edge is defined as (smaller vert id) + "#" + (larger vert id); if there is only one triangle, the second int of Eigen::Vector2i is -99



	/////////////////////////////////////////////
	// information about mesh triangles
	/////////////////////////////////////////////
	std::vector<std::vector<int>> node_triangles; // triangles that share this vertex
	std::vector<Eigen::Vector3i> triangles;
	std::vector<double> area_triangle;	
	// the followings are triangles' fundamental forms in the undeformed configuration
	std::vector<Eigen::Vector3d> normal_triangle; // triangle's normal in the undeformed configuration
	std::vector<Eigen::Matrix3d> midEdge_normal_triangle; // triangle mid-edge's normal in the undeformed configuration; each column is a normal
	std::vector<Eigen::Matrix3d> T_triangle;
	std::vector<Eigen::Matrix3d> Q_triangle;
	std::vector<Eigen::Matrix2d> l_bar_triangle;
	std::vector<double> H_triangle;
	std::vector<double> K_triangle;
	// the followings are triangles' fundamental forms in the deformed configuration
	std::vector<Eigen::Vector3d> normal_deformed_triangle; 
	std::vector<Eigen::Matrix3d> midEdge_normal_deformed_triangle;
	std::vector<Eigen::Matrix3d> t_triangle;
	std::vector<Eigen::Matrix3d> q_triangle;
	std::vector<Eigen::Matrix2d> l_triangle;
	std::vector<Eigen::Matrix3d> F0_triangle;
	std::vector<Eigen::Matrix3d> F1_triangle;
	std::vector<Eigen::Matrix3d> F2_triangle;




	// read cloth mesh from obj file
	void readMesh(std::string filePath, double thickness);

	// calculate the triangle's T, Q, a_bar, b_bar etc.
	void calUndeformedFundamentalForm();



};


// Struct of obj mesh
struct objMesh
{
	std::vector<Eigen::Vector3d> vertices;
	std::vector<std::vector<int>> faces;
};

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