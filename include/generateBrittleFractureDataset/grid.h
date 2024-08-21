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

};


#endif













