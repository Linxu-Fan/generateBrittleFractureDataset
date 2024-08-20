#ifndef PARTICLES_H

#define PARTICLES_H

#include "utils.h"


// Struct of particle type-1
struct mpmParticle {

	double mass = 0; // each particle's mass
	double volume = 0; // each particle's volume

	Eigen::Vector3d velocity = { 0 , 0 , 0 }; // each particle's velocity
	Eigen::Vector3i posIndex = { 0 , 0 , 0}; // particle base index
	Eigen::Vector3d position = { 0 , 0 , 0 }; // each particle's position 

	Eigen::MatrixXd weight;
	Eigen::MatrixXd deltaWeight;
	Eigen::Matrix3d F = Eigen::Matrix3d::Identity(); // each particle's deformation gradient
	Eigen::Matrix3d affine = Eigen::Matrix3d::Zero(); // each particle's affine term
	Eigen::Matrix3d cauchyStress = Eigen::Matrix3d::Zero(); // each particle's internal cauchy stress

	// if the mpm particle is a cloth particle, it has the following additional properties
	bool ifCloth = false;
	int corresNode_Mesh = -99; // the corresponding mesh vertex of this MPM Particle


	// node index in the support radius of this particle. The order of the vector is important
	std::vector<int> supportNodes; // store the position of the node in vector "nodeVec" 
	std::vector<double> supportNodeWeight; // store the weight of node to the particle
	std::vector<Eigen::Vector3d> supportNodeDeltaWeight; // store the weight of node to the particle


	// material 
	int material = 0; // material is stored in a vector;

	int color = 0;
	double dp = 0;

};


// initialize MPM particles from mesh
void initParticlesFromMesh(std::vector<mpmParticle>& particles, Mesh& objMesh, int materialIndex, double density);

// calculate the deformed triangle's quantities
void calDeformedTriQuantities(std::vector<mpmParticle>& particles, Mesh& objMesh);

// calculate partial 

#endif













