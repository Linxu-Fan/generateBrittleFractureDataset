#include "advance.h"


// calculate particles' weights and find neighbouring nodes
void calWeightsAndNodes(std::vector<mpmParticle>& particles, parametersSim& param, std::vector<Grid>& nodesVec)
{
	std::map<std::string, int> gridMap; // store the key and value of gridMap: active grid

	Eigen::Vector3i minCellIndex = { 100000000, 100000000, 100000000 };
	Eigen::Vector3i maxCellIndex = { -100000000, -100000000, -100000000 };
#pragma omp parallel for num_threads(param.numOfThreads)
	for (int f = 0; f < particles.size(); f++)
	{
		weightAndDreri  WD = calWeight(param.dx, particles[f].position);

		particles[f].posIndex = WD.ppIndex;
		particles[f].weight = WD.weight;
		particles[f].deltaWeight = WD.deltaWeight;

		particles[f].supportNodes.clear();
		particles[f].supportNodeWeight.clear();
		particles[f].supportNodeDeltaWeight.clear();


		minCellIndex[0] = std::min(WD.ppIndex[0], minCellIndex[0]);
		maxCellIndex[0] = std::max(WD.ppIndex[0], maxCellIndex[0]);

		minCellIndex[1] = std::min(WD.ppIndex[1], minCellIndex[1]);
		maxCellIndex[1] = std::max(WD.ppIndex[1], maxCellIndex[1]);

		minCellIndex[2] = std::min(WD.ppIndex[2], minCellIndex[2]);
		maxCellIndex[2] = std::max(WD.ppIndex[2], maxCellIndex[2]);
	};

	// find parameters that used to devide the background grid into disconnected patches
	int blockSize = 6; // the size of each block. The minimum size for quadratic kernel is 5!!!!!!!!!!!!!!!!
	std::map<std::string, std::vector<int>> bottomLeftBloack;
	std::map<std::string, std::vector<int>> bottomRightBloack;
	std::map<std::string, std::vector<int>> topLeftBloack;
	std::map<std::string, std::vector<int>> topRightBloack;
	Eigen::Vector3i span = { maxCellIndex[0] - minCellIndex[0], maxCellIndex[1] - minCellIndex[1], maxCellIndex[2] - minCellIndex[2] };
	int minSpan = std::min(span[0], std::min(span[1], span[2]));
	if (minSpan == span[0])
	{
		for (int f = 0; f < particles.size(); f++)
		{
			int remainZ = (particles[f].posIndex[2] - minCellIndex[2]) % (2 * blockSize);
			int remainY = (particles[f].posIndex[1] - minCellIndex[1]) % (2 * blockSize);

			int numZ = (particles[f].posIndex[2] - minCellIndex[2]) / (2 * blockSize);
			int numY = (particles[f].posIndex[1] - minCellIndex[1]) / (2 * blockSize);
			std::string blockID = std::to_string(numZ) + "#" + std::to_string(numY);
			if (remainZ >= 0 && remainZ < blockSize) // left 
			{
				if (remainY >= 0 && remainY < blockSize) // bottom left
				{
					bottomLeftBloack[blockID].push_back(f);
				}
				else // top left
				{
					topLeftBloack[blockID].push_back(f);
				}
			}
			else // right
			{
				if (remainY >= 0 && remainY < blockSize) // top right
				{
					bottomRightBloack[blockID].push_back(f);
				}
				else // bottm right
				{
					topRightBloack[blockID].push_back(f);
				}
			}
		}
	}
	else if (minSpan == span[1])
	{
		for (int f = 0; f < particles.size(); f++)
		{
			int remainX = (particles[f].posIndex[0] - minCellIndex[0]) % (2 * blockSize);
			int remainZ = (particles[f].posIndex[2] - minCellIndex[2]) % (2 * blockSize);

			int numX = (particles[f].posIndex[0] - minCellIndex[0]) / (2 * blockSize);
			int numZ = (particles[f].posIndex[2] - minCellIndex[2]) / (2 * blockSize);
			std::string blockID = std::to_string(numX) + "#" + std::to_string(numZ);
			if (remainX >= 0 && remainX < blockSize) // left 
			{
				if (remainZ >= 0 && remainZ < blockSize) // bottom left
				{
					bottomLeftBloack[blockID].push_back(f);
				}
				else // top left
				{
					topLeftBloack[blockID].push_back(f);
				}
			}
			else // right
			{
				if (remainZ >= 0 && remainZ < blockSize) // top right
				{
					bottomRightBloack[blockID].push_back(f);
				}
				else // bottm right
				{
					topRightBloack[blockID].push_back(f);
				}
			}
		}
	}
	else
	{
		for (int f = 0; f < particles.size(); f++)
		{
			int remainX = (particles[f].posIndex[0] - minCellIndex[0]) % (2 * blockSize);
			int remainY = (particles[f].posIndex[1] - minCellIndex[1]) % (2 * blockSize);

			int numX = (particles[f].posIndex[0] - minCellIndex[0]) / (2 * blockSize);
			int numY = (particles[f].posIndex[1] - minCellIndex[1]) / (2 * blockSize);
			std::string blockID = std::to_string(numX) + "#" + std::to_string(numY);
			if (remainX >= 0 && remainX < blockSize) // left 
			{
				if (remainY >= 0 && remainY < blockSize) // bottom left
				{
					bottomLeftBloack[blockID].push_back(f);
				}
				else // top left
				{
					topLeftBloack[blockID].push_back(f);
				}
			}
			else // right
			{
				if (remainY >= 0 && remainY < blockSize) // bottom right
				{
					bottomRightBloack[blockID].push_back(f);
				}
				else // top right
				{
					topRightBloack[blockID].push_back(f);
				}
			}
		}



	}


	int count = nodesVec.size() - 1; // count the number of active grid node
	for (int f = 0; f < particles.size(); f++)
	{
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				for (int k = 0; k < 3; k++)
				{
					std::string ID = calculateID_string(particles[f].posIndex[0] + i, particles[f].posIndex[1] + j, particles[f].posIndex[2] + k);

					if (gridMap.find(ID) == gridMap.end())
					{
						count += 1;
						gridMap[ID] = count;

						Grid node;
						nodesVec.push_back(node);
						nodesVec[count].posIndex = { particles[f].posIndex[0] + i , particles[f].posIndex[1] + j , particles[f].posIndex[2] + k };
						nodesVec[count].position = nodesVec[count].posIndex.cast<double>() * param.dx;

					}

				};
			};
		};

	}


	std::vector<std::vector<int>> bottomLeftBloackVec;
	std::map<std::string, std::vector<int>>::iterator it;
	for (it = bottomLeftBloack.begin(); it != bottomLeftBloack.end(); it++)
	{
		bottomLeftBloackVec.push_back(it->second);
	}
	std::vector<std::vector<int>> topLeftBloackVec;
	for (it = topLeftBloack.begin(); it != topLeftBloack.end(); it++)
	{
		topLeftBloackVec.push_back(it->second);
	}
	std::vector<std::vector<int>> topRightBloackVec;
	for (it = topRightBloack.begin(); it != topRightBloack.end(); it++)
	{
		topRightBloackVec.push_back(it->second);
	}
	std::vector<std::vector<int>> bottomRightBloackVec;
	for (it = bottomRightBloack.begin(); it != bottomRightBloack.end(); it++)
	{
		bottomRightBloackVec.push_back(it->second);
	}


#pragma omp parallel for num_threads(param.numOfThreads)
	for (int n = 0; n < bottomLeftBloackVec.size(); n++)
	{
		for (int m = 0; m < bottomLeftBloackVec[n].size(); m++)
		{
			int f = bottomLeftBloackVec[n][m];
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					for (int k = 0; k < 3; k++)
					{
						std::string ID = calculateID_string(particles[f].posIndex[0] + i, particles[f].posIndex[1] + j, particles[f].posIndex[2] + k);
						double weight = particles[f].weight(0, i) * particles[f].weight(1, j) * particles[f].weight(2, k);
						Eigen::Vector3d deltaWeight = { particles[f].deltaWeight(0, i) * particles[f].weight(1, j) * particles[f].weight(2, k),   particles[f].weight(0, i) * particles[f].deltaWeight(1, j) * particles[f].weight(2, k),  particles[f].weight(0, i) * particles[f].weight(1, j) * particles[f].deltaWeight(2, k) };

						int nodeIndex = gridMap[ID];

						nodesVec[nodeIndex].supportParticles.push_back(f);
						nodesVec[nodeIndex].supportParticlesWeight.push_back(weight);
						nodesVec[nodeIndex].supportParticlesDeltaWeight.push_back(deltaWeight);

						particles[f].supportNodes.push_back(nodeIndex);
						particles[f].supportNodeWeight.push_back(weight);
						particles[f].supportNodeDeltaWeight.push_back(deltaWeight);

					};
				};
			};
		}
	}


#pragma omp parallel for num_threads(param.numOfThreads)
	for (int n = 0; n < topLeftBloackVec.size(); n++)
	{
		for (int m = 0; m < topLeftBloackVec[n].size(); m++)
		{
			int f = topLeftBloackVec[n][m];
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					for (int k = 0; k < 3; k++)
					{
						std::string ID = calculateID_string(particles[f].posIndex[0] + i, particles[f].posIndex[1] + j, particles[f].posIndex[2] + k);
						double weight = particles[f].weight(0, i) * particles[f].weight(1, j) * particles[f].weight(2, k);
						Eigen::Vector3d deltaWeight = { particles[f].deltaWeight(0, i) * particles[f].weight(1, j) * particles[f].weight(2, k),   particles[f].weight(0, i) * particles[f].deltaWeight(1, j) * particles[f].weight(2, k),  particles[f].weight(0, i) * particles[f].weight(1, j) * particles[f].deltaWeight(2, k) };

						int nodeIndex = gridMap[ID];

						nodesVec[nodeIndex].supportParticles.push_back(f);
						nodesVec[nodeIndex].supportParticlesWeight.push_back(weight);
						nodesVec[nodeIndex].supportParticlesDeltaWeight.push_back(deltaWeight);

						particles[f].supportNodes.push_back(nodeIndex);
						particles[f].supportNodeWeight.push_back(weight);
						particles[f].supportNodeDeltaWeight.push_back(deltaWeight);

					};
				};
			};
		}
	}


#pragma omp parallel for num_threads(param.numOfThreads)
	for (int n = 0; n < topRightBloackVec.size(); n++)
	{
		for (int m = 0; m < topRightBloackVec[n].size(); m++)
		{
			int f = topRightBloackVec[n][m];
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					for (int k = 0; k < 3; k++)
					{
						std::string ID = calculateID_string(particles[f].posIndex[0] + i, particles[f].posIndex[1] + j, particles[f].posIndex[2] + k);
						double weight = particles[f].weight(0, i) * particles[f].weight(1, j) * particles[f].weight(2, k);
						Eigen::Vector3d deltaWeight = { particles[f].deltaWeight(0, i) * particles[f].weight(1, j) * particles[f].weight(2, k),   particles[f].weight(0, i) * particles[f].deltaWeight(1, j) * particles[f].weight(2, k),  particles[f].weight(0, i) * particles[f].weight(1, j) * particles[f].deltaWeight(2, k) };

						int nodeIndex = gridMap[ID];

						Eigen::Vector3d pos = nodesVec[nodeIndex].position;
						nodesVec[nodeIndex].supportParticles.push_back(f);
						nodesVec[nodeIndex].supportParticlesWeight.push_back(weight);
						nodesVec[nodeIndex].supportParticlesDeltaWeight.push_back(deltaWeight);

						particles[f].supportNodes.push_back(nodeIndex);
						particles[f].supportNodeWeight.push_back(weight);
						particles[f].supportNodeDeltaWeight.push_back(deltaWeight);

					};
				};
			};
		}
	}


#pragma omp parallel for num_threads(param.numOfThreads)
	for (int n = 0; n < bottomRightBloackVec.size(); n++)
	{
		for (int m = 0; m < bottomRightBloackVec[n].size(); m++)
		{
			int f = bottomRightBloackVec[n][m];
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					for (int k = 0; k < 3; k++)
					{
						std::string ID = calculateID_string(particles[f].posIndex[0] + i, particles[f].posIndex[1] + j, particles[f].posIndex[2] + k);
						double weight = particles[f].weight(0, i) * particles[f].weight(1, j) * particles[f].weight(2, k);
						Eigen::Vector3d deltaWeight = { particles[f].deltaWeight(0, i) * particles[f].weight(1, j) * particles[f].weight(2, k),   particles[f].weight(0, i) * particles[f].deltaWeight(1, j) * particles[f].weight(2, k),  particles[f].weight(0, i) * particles[f].weight(1, j) * particles[f].deltaWeight(2, k) };


						int nodeIndex = gridMap[ID];

						nodesVec[nodeIndex].supportParticles.push_back(f);
						nodesVec[nodeIndex].supportParticlesWeight.push_back(weight);
						nodesVec[nodeIndex].supportParticlesDeltaWeight.push_back(deltaWeight);

						particles[f].supportNodes.push_back(nodeIndex);
						particles[f].supportNodeWeight.push_back(weight);
						particles[f].supportNodeDeltaWeight.push_back(deltaWeight);

					};
				};
			};
		}
	}


}


// particle to grid transfer
void particle2Grid(std::vector<mpmParticle>& particles, parametersSim& param, std::vector<Grid>& nodesVec)
{
	// transfer particle's mass and momentum to grid nodes
#pragma omp parallel for num_threads(param.numOfThreads)
	for (int g = 0; g < nodesVec.size(); g++)
	{
		for (int p = 0; p < nodesVec[g].supportParticles.size(); p++)
		{
			int parPosInParticleVec = nodesVec[g].supportParticles[p];
			double weight = nodesVec[g].supportParticlesWeight[p];

			double mass = particles[parPosInParticleVec].mass;
			Eigen::Vector3d modVelocity  = particles[parPosInParticleVec].velocity + particles[parPosInParticleVec].affine * (nodesVec[g].position - particles[parPosInParticleVec].position);
			nodesVec[g].mass += mass * weight;
			nodesVec[g].momentum += mass * weight * modVelocity;
		}
	}

}


// update each particle's cauchy stress
void updateParInternalForce(std::vector<mpmParticle>& particles, parametersSim& param, std::vector<Material>& particleMaterial)
{
	// calculate each particle's internal cauchy stress
#pragma omp parallel for num_threads(param.numOfThreads)
	for (int f = 0; f < particles.size(); f++)
	{
		Eigen::Matrix3d F = particles[f].F;
		double J = particles[f].F.determinant();
		int materialIndex = particles[f].material;
		Eigen::Matrix3d cauchyStressE = (particleMaterial[materialIndex].lambda * log(J) / J - particleMaterial[materialIndex].mu / J) * Eigen::Matrix3d::Identity() + particleMaterial[materialIndex].mu / J * F * F.transpose();
		



		if (particles[f].breakable == true)
		{
			Eigen::EigenSolver<Eigen::MatrixXd> es(cauchyStressE);
			Eigen::Vector3d eigenValues = { es.eigenvalues()[0].real(), es.eigenvalues()[1].real(), es.eigenvalues()[2].real() };
			Eigen::Matrix3d eigenVectors;
			eigenVectors << es.eigenvectors().real();
			double maxEigenValue = std::max(std::max(eigenValues[0], eigenValues[1]), eigenValues[2]);
			if (maxEigenValue > particleMaterial[materialIndex].thetaf)
			{
				double tempDp = (1 + particleMaterial[materialIndex].Hs) * (1 - particleMaterial[materialIndex].thetaf / maxEigenValue);
				if (tempDp >= particleMaterial[materialIndex].damageThreshold)
				{
					double expDp = 2 / (1 + 1 / (exp(particleMaterial[materialIndex].sigmoidK * tempDp))) - 1;
					if (expDp > particles[f].dp)
					{
						particles[f].dp = expDp;
					};
				}
				else
				{
					if (tempDp > particles[f].dp)
					{
						particles[f].dp = tempDp;
					};
				};
			};


			Eigen::Vector3d sigmaPlus = { 0, 0, 0 };
			for (int i = 0; i < 3; i++) {
				if (eigenValues[i] > 0) {

					if (particles[f].dp >= particleMaterial[materialIndex].damageThreshold) 
					{
						sigmaPlus[i] = 0;
					}
					else {
						sigmaPlus[i] = (1 - particles[f].dp) * eigenValues[i];
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

			cauchyStressE = sigma;
		}
		

		particles[f].cauchyStress = cauchyStressE;

	}


}


// calculate the grid node's internal force induced by particles
void calculateNodeForce(std::vector<mpmParticle>& particles, parametersSim& param, std::vector<Grid>& nodesVec)
{
	// transfer particle's interal force to grid nodes
#pragma omp parallel for num_threads(param.numOfThreads)
	for (int g = 0; g < nodesVec.size(); g++)
	{
		for (int p = 0; p < nodesVec[g].supportParticles.size(); p++)
		{
			int parPosInParticleVec = nodesVec[g].supportParticles[p];
			double weight = nodesVec[g].supportParticlesWeight[p];
			{
				// // APIC-MPM implementation
				// nodesVec[g].force += -weight * param.dt * particles[parPosInParticleVec].volume * (particles[parPosInParticleVec].F).determinant() * particles[parPosInParticleVec].cauchyStress
				// 	* nodesVec[g].supportParticlesDeltaWeight[p];

				//MLS-MPM implementation
				nodesVec[g].force += -weight / param.DP * param.dt * particles[parPosInParticleVec].volume * (particles[parPosInParticleVec].F).determinant() * (particles[parPosInParticleVec].cauchyStress * (nodesVec[g].position - particles[parPosInParticleVec].position)).transpose();
			}
		}		
	}

}


// grid momentum update
void gridUpdate(std::vector<Grid>& nodesVec, parametersSim& param)
{
	// calculate nodes' force, solve the momentum equation and update node's velocity
	// add gravity and boundary condition
#pragma omp parallel for num_threads(param.numOfThreads)
	for (int g = 0; g < nodesVec.size(); g++)
	{
		if (nodesVec[g].mass > 0)
		{
			Eigen::Vector3d velocity = nodesVec[g].momentum / nodesVec[g].mass; // node velcity of timestep n
			Eigen::Vector3d acceleration = nodesVec[g].force / nodesVec[g].mass + param.gravity;		
			nodesVec[g].velocity = velocity + param.dt * acceleration; // node velocity of Lagrangian phase
			//nodesVec[g].positionUpdated = nodesVec[g].position + nodesVec[g].velocity * param.dt;
		};
	};

}


// grid to particle transfer
void grid2Particle(std::vector<mpmParticle>& particles, parametersSim& param, std::vector<Grid>& nodesVec)
{

#pragma omp parallel for num_threads(param.numOfThreads)
	for (int f = 0; f < particles.size(); f++)
	{
		Eigen::Vector3d velocity_old = particles[f].velocity;
		particles[f].velocity = Eigen::Vector3d::Zero(); // Particle-in-cell method
		Eigen::Matrix3d affine = Eigen::Matrix3d::Zero();
		for (int g = 0; g < particles[f].supportNodes.size(); g++)
		{
			int nodePosInNodesVec = particles[f].supportNodes[g];
			Eigen::Vector3d CMultiPos = nodesVec[nodePosInNodesVec].position - particles[f].position;

			double weight = particles[f].supportNodeWeight[g];
			particles[f].velocity += weight * nodesVec[nodePosInNodesVec].velocity;
			affine += weight / param.DP * nodesVec[nodePosInNodesVec].velocity * CMultiPos.transpose();
		}
		particles[f].F = (Eigen::Matrix3d::Identity() + param.dt * affine) * particles[f].F;		
		particles[f].affine = affine;
		//particles[f].affine = (1 - param.nu) * affinePrime + param.nu / 2.0 * (affinePrime - affinePrime.transpose());

		if (particles[f].color != 0)
		{
			particles[f].velocity = velocity_old;
		}	
		particles[f].position += param.dt * particles[f].velocity;
		
	};

}


void advanceStep(std::vector<mpmParticle>& particles, parametersSim& param, std::vector<Material>& particleMaterial, int timestep) // prticle vector, timestep
{
	// // initialize background grid nodes
	std::vector<Grid> nodesVec;
	// calculate the reationreship between particles and background grid 
	calWeightsAndNodes(particles, param, nodesVec);
	// transfer information from particle to grdi nodes
	particle2Grid(particles, param, nodesVec);
	// update each material particle's cauchy stress
	updateParInternalForce(particles, param, particleMaterial);
	// calculate the grid node's internal force induced by particles
	calculateNodeForce(particles, param, nodesVec);
	// grid nodes momentum update
	gridUpdate(nodesVec, param);
	// transfer information back form grid to particles
	grid2Particle(particles, param, nodesVec);

};


