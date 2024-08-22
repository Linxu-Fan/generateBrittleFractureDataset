#ifndef DAMAGEGRADIENT_H

#define DAMAGEGRADIENT_H

#include <cmath>

#include "particles.h"
#include "utils.h"
#include "weights.h"
#include "grid.h"

// calculate the damage gradient of all particles and grid nodes.
void calDamageGradient(std::vector<mpmParticle>*, parametersSim, double, std::map<int, int>*, std::vector<Grid>*);

// calculate the damage gradient of any give point
Eigen::Vector3d calDamageGradientPoint(Eigen::Vector3d, parametersSim, double, std::map<int, int>*, std::vector<Grid>*);

double calDamageValuePoint(Eigen::Vector3d pos, parametersSim param, double dx, std::map<int, int>* gridMap, std::vector<Grid>* nodesVec);

#endif
