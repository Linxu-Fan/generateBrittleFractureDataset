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


	// node index in the support radius of this particle. The order of the vector is important
	std::vector<int> supportNodes; // store the position of the node in vector "nodeVec" 
	std::vector<double> supportNodeWeight; // store the weight of node to the particle
	std::vector<Eigen::Vector3d> supportNodeDeltaWeight; // store the weight of node to the particle


	// material 
	int material = 0; // material is stored in a vector;
	bool breakable = true;
	int color = 0;
	double dp = 0;

	Eigen::Vector3d deltaD = { 0 , 0 , 0 }; // particle's damage gradient
	double Dg = 0;

	int nearestPoint = -1000;
	Eigen::Vector3d crackSurfaceNormal = { 0, 0, 0 };


};


// initialize MPM particles from mesh
void readBunnyMeshToParticles(std::vector<mpmParticle>& particles, objMesh& bunnyMesh, int materialIndex, double density, double voume, bool breakable, Eigen::Vector3d translation, Eigen::Vector3d velocity);


void readObjMesh(std::string meshPath, std::string meshParPath, objMesh& meshSurf, objMesh& meshPars, double size);


void readSphereObjMesh(std::string meshSpherePath, objMesh& meshSphere, double size);

#endif













