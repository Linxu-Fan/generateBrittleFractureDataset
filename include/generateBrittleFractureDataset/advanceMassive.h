#ifndef ADVANCEMASSIVE_H

#define ADVANCEMASSIVE_H

#include "particles.h"
#include "grid.h"
#include "weights.h"
#include "extractCrack.h"
#include "utils.h"


// the calculation of each timestep
void advanceStepMassive(parametersSim param, std::vector<mpmParticle>* particles, int currentTimeStep, std::vector<double>& damageRatioVec, std::map<int, Eigen::Vector3d>& gridMapCrackNormal, Material& mat1); // simulation parameters, particle vector, number of timestep, ratio of the number of fully damaged particle to that of all particles, force scale factor

// calculate particles' weights and find neighbouring nodes
void calWeightsAndNodesMassive(std::vector<mpmParticle>* particles, parametersSim param, std::map<int, int>* gridMap, std::vector<Grid>* nodesVec);

matrixAndTwoValues damageField_Rankine(Eigen::Matrix3d F, parametersSim param, double dp, double dg, int timeStep, Material& mat1, bool breakable);

void findNearestCrackPointsMassive(std::vector<Grid>* grid, std::map<int, Eigen::Vector3d> gridMapCrackNormal, std::vector<mpmParticle>* particles, parametersSim param, meshObjFormat crackSurface);

// calculate particles' weights and find neighbouring nodes with crack
void  calWeightsAndNodesWithCrackMassive(std::vector<mpmParticle>* particles, parametersSim param, std::map<int, int>* gridMap, std::vector<Grid>* nodesVec, std::vector<Grid>* nodesVecCrack, std::map<int, int>* gridMapCrack);

// calculate two normal vectors of a 3D vector
std::pair<Eigen::Vector3d, Eigen::Vector3d> calculateNormalVectorsMassive(Eigen::Vector3d vectorInput);


#endif













