#ifndef GRID_H

#define GRID_H

#include <vector>
#include <math.h>
#include <Eigen/Core>

// Struct of grid
struct Grid {

	double mass = 0; // each node's mass
	Eigen::Vector3d momentum = { 0 , 0 , 0}; // each node's momentum
	Eigen::Vector3d velocity = { 0 , 0 , 0}; // each node's velocity
	Eigen::Vector3d force = { 0 , 0 , 0}; // each node's force


	Eigen::Vector3i posIndex = { 0 , 0 , 0}; // node's position in the grid
	Eigen::Vector3d position = {0 , 0 , 0}; // node's real coordinate
	Eigen::Vector3d positionUpdated = {0 , 0 , 0}; // node's updated coordinate


	// particle index in the support radius of this node. The order of the vector is important
	std::vector<int> supportParticles; // store the position of the particle in vector "particles"; int is the position of the individual particle vector 
	std::vector<double> supportParticlesWeight; // store the weight of particle to the grid node 
	std::vector<Eigen::Vector3d> supportParticlesDeltaWeight; // store the weight of node to the particle


	// if this node contacins two kinds of particles
	Eigen::Vector2i twoPoints = {-99 , -99};

	double Di = 0; // value of damage field
	double sw = 0; // sum of particle-grid weight
	Eigen::Vector3d deltaDi = { 0, 0, 0 };


	// set of crack surface points withing the grid cell
	std::vector<int> crackPoints;
	int nearestPoint = -1000; // (nearestPoint < 0) means it is far away from the crack surface
	Eigen::Vector3d crackSurfaceNormal = { 0,0,0 }; // the vector pointing from the nearest point on the crack surface to the grid node

	// parameters of contact algorithm
	double mass_0 = 0;
	Eigen::Vector3d mom_0 = { 0 , 0 , 0 };
	Eigen::Vector3d velocity_0 = { 0 , 0 , 0 };
	Eigen::Vector3d force_0 = { 0 , 0 , 0 };

	double mass_1 = 0;
	Eigen::Vector3d mom_1 = { 0 , 0 , 0 };
	Eigen::Vector3d velocity_1 = { 0 , 0 , 0 };
	Eigen::Vector3d force_1 = { 0 , 0 , 0 };

};


#endif













