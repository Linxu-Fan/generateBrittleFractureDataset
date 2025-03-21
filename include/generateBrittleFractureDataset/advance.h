﻿#ifndef ADVANCE_H

#define ADVANCE_H

#include "particles.h"
#include "grid.h"
#include "weights.h"
#include "extractCrack.h"


// find the surrounding support nodes of each particle and calculate the weights of the particle
void calWeightsAndNodes(std::vector<mpmParticle>& particles, parametersSim& param, std::vector<Grid>& nodesVec, std::map<std::string, int>& gridMap);


// particle to grid transfer
void particle2Grid(std::vector<mpmParticle>& particles, parametersSim& param, std::vector<Grid>& nodesVec);


// update each particle's cauchy stress
void updateParInternalForce(std::vector<mpmParticle>& particles, parametersSim& param, std::vector<Material>& partclieMaterial);


// calculate the grid node's internal force induced by particles
void calculateNodeForce(std::vector<mpmParticle>& particles, parametersSim& param, std::vector<Grid>& nodesVec);


// particle to grid transfer
void gridUpdate(std::vector<Grid>& nodesVec, parametersSim& param );


// grid to particle transfer
void grid2Particle(std::vector<mpmParticle>& particles, parametersSim& param, std::vector<Grid>& nodesVec);


// apply point force
void applyPointForce(parametersSim& param, std::vector<Grid>& nodesVec, std::map<std::string, int>& gridMap);


// extract crack surface
std::tuple<bool, meshObjFormat, meshObjFormat, std::vector<meshObjFormat>> tryToExtractCracks(std::vector<mpmParticle>& particles, parametersSim& param, int timestep);


// the calculation of each timestep
void advanceStep(std::vector<mpmParticle>& particles, parametersSim& param, std::vector<Material>& partclieMaterial, int timestep); // simulation parameters, particle vector, the total number of type-1,2,3 particles



#endif













