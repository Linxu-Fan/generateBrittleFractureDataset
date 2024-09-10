#include "advanceMassive.h"




// calculate particles' weights and find neighbouring nodes
void calWeightsAndNodesMassive(std::vector<mpmParticle>* particles, parametersSim param, std::map<int, int>* gridMap, std::vector<Grid>* nodesVec)
{
#pragma omp parallel for num_threads(param.numOfThreads)
    for (int f = 0; f < particles->size(); f++) {
        struct weightAndDreri WD = calWeight(param.dx, (*particles)[f].position);

        //cout<< (*particles)[f].pos[0]
        (*particles)[f].posIndex = WD.ppIndex;
        (*particles)[f].weight = WD.weight;
        (*particles)[f].deltaWeight = WD.deltaWeight;


        (*particles)[f].supportNodes.clear();
        (*particles)[f].supportNodeWeight.clear();
    };

    int count = -1; // count the number of active grid node
    // number of grid nodes per edge


    // calculate node damage value
    for (int f = 0; f < particles->size(); f++) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    int ID = calculateID((*particles)[f].posIndex[0] + i, (*particles)[f].posIndex[1] + j, (*particles)[f].posIndex[2] + k, param.length, param.dx);
                    double weight = (*particles)[f].weight(0, i) * (*particles)[f].weight(1, j) * (*particles)[f].weight(2, k);

                    if (weight != 0)
                    {
                        if ((*gridMap).find(ID) == (*gridMap).end())
                        {
                            count += 1;
                            (*gridMap)[ID] = count;

                            Grid node;
                            (*nodesVec).push_back(node);

                            (*nodesVec)[count].posIndex = { (*particles)[f].posIndex[0] + i, (*particles)[f].posIndex[1] + j, (*particles)[f].posIndex[2] + k };



                            (*nodesVec)[count].supportParticles.push_back(f);
                            (*nodesVec)[count].supportParticlesWeight.push_back(weight);

                            (*particles)[f].supportNodes.push_back(count);
                            (*particles)[f].supportNodeWeight.push_back(weight);


                        }
                        else
                        {
                            int nodeIndex = (*gridMap)[ID];

                            (*nodesVec)[nodeIndex].supportParticles.push_back(f);
                            (*nodesVec)[nodeIndex].supportParticlesWeight.push_back(weight);

                            (*particles)[f].supportNodes.push_back(nodeIndex);
                            (*particles)[f].supportNodeWeight.push_back(weight);
                        }

                    };
                };
            };
        };
    };


}



void findNearestCrackPointsMassive(std::vector<Grid>* grid, std::map<int, Eigen::Vector3d> gridMapCrackNormal, std::vector<mpmParticle>* particles, parametersSim param, meshObjFormat crackSurface)
{
    for (int i = 0; i < crackSurface.faces.size(); i++)
    {
        int p0 = crackSurface.faces[i][0];
        int p1 = crackSurface.faces[i][1];
        int p2 = crackSurface.faces[i][2];

        Eigen::Vector3d v0 = crackSurface.vertices[p0];
        Eigen::Vector3d v1 = crackSurface.vertices[p0];
        Eigen::Vector3d v2 = crackSurface.vertices[p0];

        Eigen::Vector3d nor = (v1 - v0).cross(v2 - v0).normalized();
        crackSurface.facesNormal.push_back(nor);
    }


    gridMapCrackNormal.clear();
    // find grid node's normal vector
    // can't use multi thread
    for (int f = 0; f < grid->size(); f++)
    {

        (*grid)[f].nearestPoint = -1000;
        (*grid)[f].crackSurfaceNormal = { 0,0,0 };

        Eigen::Vector3d pos = (*grid)[f].posIndex.cast<double>() * param.dx;
        double minDistance = 1.0E10;
        Eigen::Vector3d crackNormal;
        for (int i = 0; i < crackSurface.faces.size(); i++)
        {
            Eigen::Vector3d pointOnFace = crackSurface.vertices[crackSurface.faces[i][0]];
            double rawDistance = (pointOnFace - pos).dot(crackSurface.facesNormal[i]);
            double distance = std::abs(rawDistance);
            if (distance < minDistance)
            {
                minDistance = distance;
                crackNormal = (rawDistance * crackSurface.facesNormal[i]).normalized();
            }
        }
        if (minDistance <= param.dx * 4.0)
        {
            (*grid)[f].crackSurfaceNormal = crackNormal;
            int ID = calculateID((*grid)[f].posIndex[0], (*grid)[f].posIndex[1], (*grid)[f].posIndex[2], param.length, param.dx);
            gridMapCrackNormal[ID] = crackNormal;
        }

    }


    // find a particle's normal vector
#pragma omp parallel for num_threads(param.numOfThreads)
    for (int f = 0; f < particles->size(); f++)
    {
        (*particles)[f].nearestPoint = -1000;
        (*particles)[f].crackSurfaceNormal = { 0,0,0 };

        Eigen::Vector3d pos = (*particles)[f].position;
        double minDistance = 1.0E10;
        Eigen::Vector3d crackNormal;
        for (int i = 0; i < crackSurface.faces.size(); i++)
        {
            Eigen::Vector3d pointOnFace = crackSurface.vertices[crackSurface.faces[i][0]];
            double rawDistance = (pointOnFace - pos).dot(crackSurface.facesNormal[i]);
            double distance = std::abs(rawDistance);
            if (distance < minDistance)
            {
                minDistance = distance;
                crackNormal = (rawDistance * crackSurface.facesNormal[i]).normalized();
            }
        }
        if (minDistance <= param.dx * 1.5)
        {
            (*particles)[f].nearestPoint = 1000;
            (*particles)[f].crackSurfaceNormal = crackNormal;
        }

    }




}




// update cauchy stress and damage value using damage field method by Rankine model
matrixAndTwoValues damageField_Rankine(Eigen::Matrix3d F, parametersSim param, double dp, double dg, int timeStep, Material& mat1, bool breakable)
{
    //// Compressible neo-hooKean model
    double J = F.determinant();
    Eigen::Matrix3d cauchyStressE = (mat1.lambda * log(J) / J - mat1.mu / J) * Eigen::Matrix3d::Identity() + mat1.mu / J * F * F.transpose();

    // compute eigenvalue and eigenvector
    Eigen::EigenSolver<Eigen::MatrixXd> es(cauchyStressE);
    Eigen::Vector3d eigenValues = { es.eigenvalues()[0].real(), es.eigenvalues()[1].real(), es.eigenvalues()[2].real() };
    Eigen::Matrix3d eigenVectors;
    eigenVectors << es.eigenvectors().real();
    double maxEigenValue = std::max(std::max(eigenValues[0], eigenValues[1]), eigenValues[2]);

    if (maxEigenValue > mat1.thetaf) {

        double tempDp = (1 + mat1.Hs) * (1 - mat1.thetaf / maxEigenValue);
        if (param.damageThreshold == 1.0) {
            if (maxEigenValue > (1 + 1 / mat1.Hs) * mat1.thetaf) {
                dp = 1;
                if (dg == 0)
                {
                    dg = (double)timeStep;
                }
            }
            else {
                if (tempDp > dp) {
                    dp = tempDp;
                };
            };
        }
        else {
            if (tempDp >= param.damageThreshold) {
                if (dg == 0)
                {
                    dg = (double)timeStep;
                }

                double expDp = 2 / (1 + 1 / (exp(param.sigmoidK * tempDp))) - 1;
                if (expDp > dp) {
                    dp = expDp;
                };
            }
            else {
                if (tempDp > dp) {
                    dp = tempDp;
                };
            };
        };
    };

    Eigen::Vector3d sigmaPlus = { 0, 0, 0 };
    for (int i = 0; i < 3; i++) {
        if (eigenValues[i] > 0) {

            if (dp >= param.damageThreshold) {
                sigmaPlus[i] = 0;
            }
            else {
                sigmaPlus[i] = (1 - dp) * eigenValues[i];
            };

        }
        else {
            sigmaPlus[i] = eigenValues[i];
        };
    };

    Eigen::Matrix3d sigma = Eigen::Matrix3d::Zero();
    for (int i = 0; i < 3; i++) {
        sigma = sigma + sigmaPlus[i] * eigenVectors.col(i) * (eigenVectors.col(i).transpose());
    };

    if (breakable)
    {
        struct matrixAndTwoValues res(sigma, dp, dg);
        return res;
    }
    else
    {
        struct matrixAndTwoValues res(cauchyStressE, 0, 0);
        return res;
    }

   
};


// calculate particles' weights and find neighbouring nodes with crack
void  calWeightsAndNodesWithCrackMassive(std::vector<mpmParticle>* particles, parametersSim param, std::map<int, int>* gridMap, std::vector<Grid>* nodesVec, std::vector<Grid>* nodesVecCrack, std::map<int, int>* gridMapCrack)
{
    for (int f = 0; f < particles->size(); f++)
    {
        struct weightAndDreri  WD = calWeight(param.dx, (*particles)[f].position);

        (*particles)[f].posIndex = WD.ppIndex;
        (*particles)[f].weight = WD.weight;
        (*particles)[f].deltaWeight = WD.deltaWeight;
    };

    int count = -1; // count the number of active grid node
    // number of grid nodes per edge

    // calculate node damage value
    for (int f = 0; f < particles->size(); f++)
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    int ID = calculateID((*particles)[f].posIndex[0] + i, (*particles)[f].posIndex[1] + j, (*particles)[f].posIndex[2] + k, param.length, param.dx);
                    double weight = (*particles)[f].weight(0, i) * (*particles)[f].weight(1, j) * (*particles)[f].weight(2, k);


                    if (weight != 0)
                    {
                        if ((*gridMap).find(ID) == (*gridMap).end())
                        {
                            count += 1;
                            (*gridMap)[ID] = count;
                             Grid node;
                            (*nodesVec).push_back(node);
                            (*nodesVec)[count].posIndex = { (*particles)[f].posIndex[0] + i , (*particles)[f].posIndex[1] + j , (*particles)[f].posIndex[2] + k };
                        }
                        else
                        {
                            int eid = (*gridMap)[ID];
                            if ((*gridMapCrack).find(ID) != (*gridMapCrack).end())
                            {
                                int eidCrack = (*gridMapCrack)[ID];
                                (*nodesVec)[eid].crackSurfaceNormal = (*nodesVecCrack)[eidCrack].crackSurfaceNormal;
                            }
                        }

                    };

                };
            };
        };

    };


}



// calculate two normal vectors of a 3D vector
std::pair<Eigen::Vector3d, Eigen::Vector3d> calculateNormalVectorsMassive(Eigen::Vector3d vectorInput)
{
    Eigen::Vector3d n0 = { 0,0,0 }, n1 = { 0,0,0 };
    if (vectorInput[2] == 0)
    {
        n0 = { 0,0,1 };
        n1 = n0.cross(vectorInput).normalized();
    }
    else
    {
        n0 = { 1 , 0 , -vectorInput[0] / vectorInput[2] };
        n0 = n0.normalized();
        n1 = n0.cross(vectorInput).normalized();
    }

    std::pair<Eigen::Vector3d, Eigen::Vector3d> result(n0, n1);
    return result;

}




// No contact algorithm
void advanceStepMassive(parametersSim param, std::vector<mpmParticle>* particles, int currentTimeStep, std::vector<double>& damageRatioVec, std::map<int, Eigen::Vector3d>& gridMapCrackNormal, Material& mat1) // prticle std::vector, timestep
{

        std::map<int, int> gridMap; // store the key and value of gridMap: active grid
        std::vector<Grid> nodesVec;


        calWeightsAndNodesMassive(particles, param, &gridMap, &nodesVec);



        if (currentTimeStep % 10 == 0 && currentTimeStep != 0)
        {
            if (damageRatioVec[damageRatioVec.size() - 1] > 0)
            {
                for (int fg = 0; fg < particles->size(); fg++)
                {
                    (*particles)[fg].nearestPoint = -1000;
                    (*particles)[fg].crackSurfaceNormal = { 0, 0 ,0 };
                }


                {
                    std::vector<extractCrackSurface::Particle> particlesRaw;
                    extractCrackSurface::parametersSim paramCrack;
                    for (int i = 0; i < particles->size(); i++)
                    {
                        if ((*particles)[i].breakable)
                        {
                            particlesRaw.push_back(extractCrackSurface::Particle((*particles)[i].position, (*particles)[i].velocity, (*particles)[i].mass, 0, (*particles)[i].dp));
                        }                 
                    }
                    paramCrack.dx = param.dx;
                    paramCrack.damageThreshold = paramCrack.damageThreshold;

                    std::tuple<bool, extractCrackSurface::meshObjFormat, extractCrackSurface::meshObjFormat, std::vector<extractCrackSurface::meshObjFormat>> cracks = extractCrackSurface::extractCrackSurf(&particlesRaw, paramCrack);
                    if (std::get<0>(cracks) && std::get<1>(cracks).faces.size() != 0)
                    {
                        meshObjFormat crackSurfacePartialCut;
                        crackSurfacePartialCut.vertices = std::get<1>(cracks).vertices;
                        crackSurfacePartialCut.faces = std::get<1>(cracks).faces;

                        findNearestCrackPointsMassive(&nodesVec, gridMapCrackNormal, particles, param, crackSurfacePartialCut);
                    }

                }

            }

        }
        else
        {
            if (gridMapCrackNormal.size() != 0)
            {
                std::map<int, Eigen::Vector3d>::iterator it;
                for (it = gridMapCrackNormal.begin(); it != gridMapCrackNormal.end(); it++)
                {
                    int gridNodeID = it->first;
                    Eigen::Vector3d normal = it->second;
                    int gridNodeIndex = gridMap[gridNodeID];
                    nodesVec[gridNodeIndex].crackSurfaceNormal = normal;
                }
            }

        }


        // transfer particle's mass and momentum to grid nodes
#pragma omp parallel for num_threads(param.numOfThreads)
        for (int g = 0; g < nodesVec.size(); g++)
        {
            for (int p = 0; p < nodesVec[g].supportParticles.size(); p++)
            {
                int parPosInParticleVec = nodesVec[g].supportParticles[p];
                double weight = nodesVec[g].supportParticlesWeight[p];

                Eigen::Vector3d CMultiPos = nodesVec[g].posIndex.cast<double>() * param.dx - ((*particles)[parPosInParticleVec].position);
                Eigen::Vector3d affineContribution = (*particles)[parPosInParticleVec].affine * CMultiPos;

                if ((*particles)[parPosInParticleVec].nearestPoint < 0)
                {
                    // transfer mass and momentum
                    nodesVec[g].mass_0 += (*particles)[parPosInParticleVec].mass * weight;
                    nodesVec[g].mom_0 += (*particles)[parPosInParticleVec].mass * weight * ((*particles)[parPosInParticleVec].velocity + affineContribution);
                }
                else
                {
                    if ((*particles)[parPosInParticleVec].crackSurfaceNormal.dot(nodesVec[g].crackSurfaceNormal) >= 0)
                    {
                        // transfer mass and momentum
                        nodesVec[g].mass_0 += (*particles)[parPosInParticleVec].mass * weight;
                        nodesVec[g].mom_0 += (*particles)[parPosInParticleVec].mass * weight * ((*particles)[parPosInParticleVec].velocity + affineContribution);
                    }
                    else
                    {
                        // transfer mass and momentum
                        nodesVec[g].mass_1 += (*particles)[parPosInParticleVec].mass * weight;
                        nodesVec[g].mom_1 += (*particles)[parPosInParticleVec].mass * weight * ((*particles)[parPosInParticleVec].velocity + affineContribution);
                    }
                }

            }

        }




        int numOfFullyDamagedParticle = 0; // the number of fully damaged particles in the domain
        // compute the actual internal forces of two velocity fields
#pragma omp parallel for num_threads(param.numOfThreads)
        for (int f = 0; f < particles->size(); f++)
        {
            // update strain softening
            matrixAndTwoValues res = damageField_Rankine((*particles)[f].F, param, (*particles)[f].dp, (*particles)[f].Dg, currentTimeStep, mat1, (*particles)[f].breakable);
            (*particles)[f].dp = res.Dp;
            (*particles)[f].Dg = res.Dg;
            (*particles)[f].cauchyStress = res.F;

            if ((*particles)[f].dp >= param.damageThreshold) {
                numOfFullyDamagedParticle += 1;
            }

        }

#pragma omp parallel for num_threads(param.numOfThreads)
        for (int g = 0; g < nodesVec.size(); g++)
        {
            for (int p = 0; p < nodesVec[g].supportParticles.size(); p++)
            {
                int parPosInParticleVec = nodesVec[g].supportParticles[p];
                double weight = nodesVec[g].supportParticlesWeight[p];

                Eigen::Vector3d CMultiPos = nodesVec[g].posIndex.cast<double>() * param.dx - ((*particles)[parPosInParticleVec].position);

                if ((*particles)[parPosInParticleVec].nearestPoint < 0)
                {
                    nodesVec[g].force_0 += -weight / param.DP * (*particles)[parPosInParticleVec].volume * ((*particles)[parPosInParticleVec].F).determinant() * ((*particles)[parPosInParticleVec].cauchyStress * CMultiPos).transpose();
                }
                else
                {
                    if ((*particles)[parPosInParticleVec].crackSurfaceNormal.dot(nodesVec[g].crackSurfaceNormal) >= 0)
                    {
                        nodesVec[g].force_0 += -weight / param.DP * (*particles)[parPosInParticleVec].volume * ((*particles)[parPosInParticleVec].F).determinant() * ((*particles)[parPosInParticleVec].cauchyStress * CMultiPos).transpose();
                    }
                    else
                    {
                        nodesVec[g].force_1 += -weight / param.DP * (*particles)[parPosInParticleVec].volume * ((*particles)[parPosInParticleVec].F).determinant() * ((*particles)[parPosInParticleVec].cauchyStress * CMultiPos).transpose();
                    }
                }

            }
        }

        double ratioOfFullyDamagedParticle = (double)numOfFullyDamagedParticle / (double)particles->size(); // the ratio of the number of fully damaged particle to that of all particles
        damageRatioVec.push_back(ratioOfFullyDamagedParticle);


        // if the damage ratio doesn't change for more than 10 timesteps, then stop;
        if (damageRatioVec.size() > 50) {
            double sumRatio = 0;
            for (int ki = damageRatioVec.size() - 50; ki < damageRatioVec.size(); ki++) {
                sumRatio += damageRatioVec[ki];
            }
        }


        // apply rigid body force
        for (int f = 0; f < param.appliedForce.size(); f++) 
        {
            Eigen::Vector3d forceMagnitude = param.appliedForce[f].first;
            Eigen::Vector3d forcePosition = param.appliedForce[f].second;



            struct weightAndDreri WD = calWeight(param.dx, forcePosition);
            Eigen::Vector3i ppIndex = WD.ppIndex;
            Eigen::MatrixXd weightForce = WD.weight;

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        int ID = calculateID(ppIndex[0] + i, ppIndex[1] + j, ppIndex[2] + k, param.length, param.dx);
                        double weight = weightForce(0, i) * weightForce(1, j) * weightForce(2, k);

                        if (weight != 0) {
                            int eid = gridMap[ID];
                            nodesVec[eid].force_0 += weight * forceMagnitude;
                        };
                    };
                };
            };
        }



        // calculate nodes' force, solve the momentum equation and update node's velocity
#pragma omp parallel for num_threads(param.numOfThreads)
        for (int g = 0; g < nodesVec.size(); g++)
        {

            if (nodesVec[g].mass_0 > 0)
            {
                Eigen::Vector3d velocity = nodesVec[g].mom_0 / nodesVec[g].mass_0; // node velcity of timestep n

                Eigen::Vector3d acceleration = nodesVec[g].force_0 / nodesVec[g].mass_0;
                nodesVec[g].velocity_0 = velocity + param.dt * acceleration; // node velocity of Lagrangian phase

            };


            if (nodesVec[g].mass_1 > 0)
            {
                Eigen::Vector3d velocity = nodesVec[g].mom_1 / nodesVec[g].mass_1; // node velcity of timestep n

                Eigen::Vector3d acceleration = nodesVec[g].force_1 / nodesVec[g].mass_1;
                nodesVec[g].velocity_1 = velocity + param.dt * acceleration; // node velocity of Lagrangian phase

            }



            nodesVec[g].mass = nodesVec[g].mass_0 + nodesVec[g].mass_1;
            nodesVec[g].momentum = nodesVec[g].mass_0 * nodesVec[g].velocity_0 + nodesVec[g].mass_1 * nodesVec[g].velocity_1;
            if (nodesVec[g].mass != 0)
            {
                nodesVec[g].velocity = nodesVec[g].momentum / nodesVec[g].mass;
            }

        };


        // add contact force
        for (int g = 0; g < nodesVec.size(); g++)
        {
            if (nodesVec[g].mass_0 > 0 && nodesVec[g].mass_1 > 0)
            {

                Eigen::Vector3d field0_normal = -nodesVec[g].crackSurfaceNormal.normalized();
                Eigen::Vector3d field1_normal = -field0_normal;


                if ((nodesVec[g].velocity - nodesVec[g].velocity_0).dot(field0_normal) < 0)
                {
                    double force_normal = nodesVec[g].mass_0 / param.dt * (nodesVec[g].velocity - nodesVec[g].velocity_0).dot(field0_normal);
                    Eigen::Vector3d contact_force = force_normal * field0_normal;
                    nodesVec[g].velocity_0 += contact_force / nodesVec[g].mass_0 * param.dt;
                }

                if ((nodesVec[g].velocity - nodesVec[g].velocity_1).dot(field1_normal) < 0)
                {
                    double force_normal = nodesVec[g].mass_1 / param.dt * (nodesVec[g].velocity - nodesVec[g].velocity_1).dot(field1_normal);
                    Eigen::Vector3d contact_force = force_normal * field1_normal;
                    nodesVec[g].velocity_1 += contact_force / nodesVec[g].mass_1 * param.dt;
                }


            }

        }



        //  update particle's velocity, and position
#pragma omp parallel for num_threads(param.numOfThreads)
        for (int f = 0; f < particles->size(); f++)
        {
            (*particles)[f].velocity = (*particles)[f].velocity * 0; // Particle-in-cell method
            (*particles)[f].affine = Eigen::Matrix3d::Zero();


            for (int g = 0; g < (*particles)[f].supportNodes.size(); g++)
            {

                int nodePosInNodesVec = (*particles)[f].supportNodes[g];
                Eigen::Vector3d CMultiPos = nodesVec[nodePosInNodesVec].posIndex.cast<double>() * param.dx - (*particles)[f].position;
                double weight = (*particles)[f].supportNodeWeight[g];


                if ((*particles)[f].nearestPoint < 0)
                {
                    (*particles)[f].velocity += weight * nodesVec[nodePosInNodesVec].velocity_0;
                    (*particles)[f].affine += weight / param.DP * nodesVec[nodePosInNodesVec].velocity_0 * CMultiPos.transpose();
                }
                else
                {
                    if ((*particles)[f].crackSurfaceNormal.dot(nodesVec[nodePosInNodesVec].crackSurfaceNormal) >= 0)
                    {
                        (*particles)[f].velocity += weight * nodesVec[nodePosInNodesVec].velocity_0;
                        (*particles)[f].affine += weight / param.DP * nodesVec[nodePosInNodesVec].velocity_0 * CMultiPos.transpose();
                    }
                    else
                    {
                        (*particles)[f].velocity += weight * nodesVec[nodePosInNodesVec].velocity_1;
                        (*particles)[f].affine += weight / param.DP * nodesVec[nodePosInNodesVec].velocity_1 * CMultiPos.transpose();
                    }
                }

            }






            (*particles)[f].position += param.dt * (*particles)[f].velocity;
            (*particles)[f].F = (Eigen::Matrix3d::Identity() + param.dt * (*particles)[f].affine) * (*particles)[f].F;
        };



};

