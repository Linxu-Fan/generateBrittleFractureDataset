#include "weights.h"
#include "particles.h"
#include "grid.h"
#include "advance.h"



bool checkIfContatc(std::vector<mpmParticle>& particles, parametersSim& param)
{
	std::vector<Grid> nodesVec;
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


	std::vector<bool> blc(bottomLeftBloackVec.size());
#pragma omp parallel for num_threads(param.numOfThreads)
	for (int n = 0; n < bottomLeftBloackVec.size(); n++)
	{
		bool findContact = false;
		for (int m = 0; m < bottomLeftBloackVec[n].size() && findContact == false; m++)
		{
			int f = bottomLeftBloackVec[n][m];
			for (int i = 0; i < 3 && findContact == false; i++)
			{
				for (int j = 0; j < 3 && findContact == false; j++)
				{
					for (int k = 0; k < 3 && findContact == false; k++)
					{
						std::string ID = calculateID_string(particles[f].posIndex[0] + i, particles[f].posIndex[1] + j, particles[f].posIndex[2] + k);
						int nodeIndex = gridMap[ID];

						if (particles[f].breakable == true)
						{
							nodesVec[nodeIndex].twoPoints[0] = 1;
						}
						if (particles[f].breakable == false)
						{
							nodesVec[nodeIndex].twoPoints[1] = 1;
						}

						if (nodesVec[nodeIndex].twoPoints[0] + nodesVec[nodeIndex].twoPoints[1] > 0)
						{
							findContact = true;
							//goto fini; // jump to the label
						}
					};
				};
			};
		}
		blc[n] = findContact;
	}


	std::vector<bool> tlc(topLeftBloackVec.size());
#pragma omp parallel for num_threads(param.numOfThreads)
	for (int n = 0; n < topLeftBloackVec.size(); n++)
	{
		bool findContact = false;
		for (int m = 0; m < topLeftBloackVec[n].size() && findContact == false; m++)
		{
			int f = topLeftBloackVec[n][m];
			for (int i = 0; i < 3 && findContact == false; i++)
			{
				for (int j = 0; j < 3 && findContact == false; j++)
				{
					for (int k = 0; k < 3 && findContact == false; k++)
					{
						std::string ID = calculateID_string(particles[f].posIndex[0] + i, particles[f].posIndex[1] + j, particles[f].posIndex[2] + k);
						int nodeIndex = gridMap[ID];

						if (particles[f].breakable == true)
						{
							nodesVec[nodeIndex].twoPoints[0] = 1;
						}
						if (particles[f].breakable == false)
						{
							nodesVec[nodeIndex].twoPoints[1] = 1;
						}

						if (nodesVec[nodeIndex].twoPoints[0] + nodesVec[nodeIndex].twoPoints[1] > 0)
						{
							findContact = true;
						}


					};
				};
			};
		}
		tlc[n] = findContact;
	}


	std::vector<bool> trc(topRightBloackVec.size());
#pragma omp parallel for num_threads(param.numOfThreads)
	for (int n = 0; n < topRightBloackVec.size(); n++)
	{
		bool findContact = false;
		for (int m = 0; m < topRightBloackVec[n].size() && findContact == false; m++)
		{
			int f = topRightBloackVec[n][m];
			for (int i = 0; i < 3 && findContact == false; i++)
			{
				for (int j = 0; j < 3 && findContact == false; j++)
				{
					for (int k = 0; k < 3 && findContact == false; k++)
					{
						std::string ID = calculateID_string(particles[f].posIndex[0] + i, particles[f].posIndex[1] + j, particles[f].posIndex[2] + k);
						int nodeIndex = gridMap[ID];

						if (particles[f].breakable == true)
						{
							nodesVec[nodeIndex].twoPoints[0] = 1;
						}
						if (particles[f].breakable == false)
						{
							nodesVec[nodeIndex].twoPoints[1] = 1;
						}

						if (nodesVec[nodeIndex].twoPoints[0] + nodesVec[nodeIndex].twoPoints[1] > 0)
						{
							findContact = true;
						}

					};
				};
			};
		}
		trc[n] = findContact;
	}


	std::vector<bool> brc(bottomRightBloackVec.size());
#pragma omp parallel for num_threads(param.numOfThreads)
	for (int n = 0; n < bottomRightBloackVec.size(); n++)
	{
		bool findContact = false;

		for (int m = 0; m < bottomRightBloackVec[n].size() && findContact == false; m++)
		{
			int f = bottomRightBloackVec[n][m];
			for (int i = 0; i < 3 && findContact == false; i++)
			{
				for (int j = 0; j < 3 && findContact == false; j++)
				{
					for (int k = 0; k < 3 && findContact == false; k++)
					{
						std::string ID = calculateID_string(particles[f].posIndex[0] + i, particles[f].posIndex[1] + j, particles[f].posIndex[2] + k);
						int nodeIndex = gridMap[ID];

						if (particles[f].breakable == true)
						{
							nodesVec[nodeIndex].twoPoints[0] = 1;
						}
						if (particles[f].breakable == false)
						{
							nodesVec[nodeIndex].twoPoints[1] = 1;
						}

						if (nodesVec[nodeIndex].twoPoints[0] + nodesVec[nodeIndex].twoPoints[1] > 0)
						{
							findContact = true;
						}

					};
				};
			};
		}
		brc[n] = findContact;
	}



	std::set<int> sl(blc.begin(), blc.end());
	sl.insert(tlc.begin(), tlc.end());
	sl.insert(trc.begin(), trc.end());
	sl.insert(brc.begin(), brc.end());


	bool findContact = false;
	for (std::set<int>::iterator it = sl.begin(); it != sl.end(); it++)
	{
		if (*it == true)
		{
			findContact = true;
		}
	}



	return findContact;
}





int main()
{

	if (0)
	{
		std::ofstream outfile2("./output/squaredCloth.obj", std::ios::trunc);
		double radiusGround = 0.5;
		double dx = 0.01;
		for (int yi = 0; yi <= 100; yi++)
		{
			for (int xi = 0; xi <= 100; xi++)
			{
				double xc = (double)xi * dx, yc = (double)yi * dx;
				outfile2 << std::scientific << std::setprecision(8) << "v " << xc << " " << yc << " " << 0 << std::endl;
			}
		}

		for (int yi = 0; yi < 100; yi++)
		{
			for (int xi = 0; xi < 100; xi++)
			{
				int v0 = yi * 101 + xi, v1 = yi * 101 + xi + 1, v2 = yi * 101 + xi + 102, v3 = yi * 101 + xi + 101;
				outfile2 << "f " << v0 + 1 << " " << v1 + 1 << " " << v2 + 1 << std::endl;
				outfile2 << "f " << v0 + 1 << " " << v2 + 1 << " " << v3 + 1 << std::endl;
			}
		}
		outfile2.close();

	}
	else
	{
		int windows = 1;
		int numOfThreads;
		std::string processPath, outputPath;
		if (windows)
		{
			outputPath = ".//output";
			numOfThreads = 32;
		}
		else
		{
			outputPath = "//home//floyd//Linxu//tempOutput";
			numOfThreads = 24;
		}


		// simulation parameters
		parametersSim param;
		param.numOfThreads = numOfThreads;
		param.dt = 1.0E-2;
		param.dx = 0.01;
		param.updateDenpendecies();


		// initialize three types of materials
		Material material1;
		material1.density = 100;
		material1.E = 3.2E4;
		material1.bendingStressThreshold = 250;
		material1.lch = sqrt(3) * param.dx;
		material1.updateDenpendecies();
		std::vector<Material> partclieMaterial;
		partclieMaterial.push_back(material1);


		
		objMesh bunnyMeshSurf, bunnyMeshPars;
		readObjMesh("./input/bunny.obj", "./input/bunnyParticles.obj", bunnyMeshSurf, bunnyMeshPars, 0.95);
		objMesh sphereMeshPar;
		double sphereRadius = 0.02;
		readSphereObjMesh("./input/sphereParticles.obj", sphereMeshPar, sphereRadius);
		int numBunnySurfVerts = bunnyMeshSurf.vertices.size();

		// generate numData datasets
		int numData = 10000;
		for (int k = 0; k < numData; k++)
		{
			// read bunny particles
			std::vector<mpmParticle> particles;
			readBunnyMeshToParticles(particles, bunnyMeshPars, 0, material1.density, param.dx * param.dx * param.dx / 8.0, true, Eigen::Vector3d::Zero());


			// read sphere particles
			int sphereParSkipRatio = 10;
			{
				// Step 1: calcualte the diaplacement of shpere
				double theta = generateRandomDouble(0, PI);
				double phi = generateRandomDouble(0, 2.0 * PI);
				Eigen::Vector3d translation = { std::sin(theta) * std::cos(phi) - 0.5 , std::sin(theta) * std::sin(phi) - 0.5 , std::cos(theta) - 0.5 };
				std::vector<Eigen::Vector3d> vertSphere;
				for (int f = 0; f < sphereMeshPar.vertices.size(); f++)
				{
					if (f % sphereParSkipRatio == 0)
					{
						vertSphere.push_back(sphereMeshPar.vertices[f] + translation);
					}				
				}


				// Step 2: calculate the nearest distance to the bunny surface
				double minDis = 1.0e8;
				int bunnySurfVt = -99, sphereSurfVt = -99;
				for (int bv = 0; bv < bunnyMeshSurf.vertices.size(); bv++)
				{
					for (int sv = 0; sv < vertSphere.size(); sv++)
					{
						double dis = (bunnyMeshSurf.vertices[bv] - vertSphere[sv]).norm();
						if (dis < minDis)
						{
							minDis = dis;
							bunnySurfVt = bv;
							sphereSurfVt = sv;
						}
					}
				}
								
				// Step 3: move sphere particles to the target position
				Eigen::Vector3d dirToMove = bunnyMeshSurf.vertices[bunnySurfVt] - vertSphere[sphereSurfVt];
				dirToMove = dirToMove * 0.8;
				{					
					double minDisContact = -99.0;
					bool twoContact = false;
					for (int d = 140; d < 1000; d++)
					{
						std::cout << "d = " << d << std::endl;
						double ra = (double)d * 0.001;
						dirToMove += dirToMove.normalized() * sphereRadius * ra;
						
						std::vector<Eigen::Vector3d> vertSphereFinalTmp;
						for (int f = 0; f < vertSphere.size(); f++)
						{
							vertSphereFinalTmp.push_back(vertSphere[f] + dirToMove);
						}

						objMesh sphereMeshParTmp;
						std::vector<mpmParticle> particlesTmp = particles;
						sphereMeshParTmp.vertices = vertSphereFinalTmp;
						readBunnyMeshToParticles(particlesTmp, sphereMeshParTmp, 0, material1.density, param.dx * param.dx * param.dx / 8.0, false, Eigen::Vector3d::Zero());
						twoContact = checkIfContatc(particlesTmp, param);
						if (twoContact)
						{
							minDisContact = ra - 0.001;
							break;
						}
					}
					std::cout << "minDisContact = " << minDisContact << std::endl;

					dirToMove += dirToMove.normalized() * sphereRadius * minDisContact;
					std::vector<Eigen::Vector3d> vertSphereFinal;
					for (int f = 0; f < vertSphere.size(); f++)
					{
						vertSphereFinal.push_back(vertSphere[f] + dirToMove);
					}
					sphereMeshPar.vertices = vertSphereFinal;
					readBunnyMeshToParticles(particles, sphereMeshPar, 0, material1.density, param.dx * param.dx * param.dx / 8.0, false, Eigen::Vector3d::Zero());
					bool twoContact2 = checkIfContatc(particles, param);
					std::cout << "twoContact2 = " << twoContact << std::endl;

				}
				
				
				
				
				
				
				/*double minDisContact = -99.0;
				bool twoContact = false;
				for (int d = 1000; d > 0; d--)
				{
					std::cout << "d = " << d << std::endl;
					double ra = (double)d * 0.01;
					dirToMove -= dirToMove.normalized() * sphereRadius * ra;
					std::vector<Eigen::Vector3d> vertSphereFinalTmp;
					for (int f = 0; f < vertSphere.size(); f++)
					{
						vertSphereFinalTmp.push_back(vertSphere[f] + dirToMove);
					}

					objMesh sphereMeshParTmp;
					std::vector<mpmParticle> particlesTmp = particles;
					sphereMeshParTmp.vertices = vertSphereFinalTmp;
					readBunnyMeshToParticles(particlesTmp, sphereMeshParTmp, 0, material1.density, param.dx * param.dx * param.dx / 8.0, false, Eigen::Vector3d::Zero());
					twoContact = checkIfContatc(particlesTmp, param);
					if (twoContact)
					{
						minDisContact = ra + 0.01;
						break;
					}
				}
				std::cout << "minDisContact = " << minDisContact << std::endl;*/
				//dirToMove -= dirToMove.normalized() * sphereRadius * minDisContact;
				//std::vector<Eigen::Vector3d> vertSphereFinal;
				//for (int f = 0; f < vertSphere.size(); f++)
				//{
				//	vertSphereFinal.push_back(vertSphere[f] + dirToMove);
				//}
				//sphereMeshPar.vertices = vertSphereFinal;
				//readBunnyMeshToParticles(particles, sphereMeshPar, 0, material1.density, param.dx * param.dx * param.dx / 8.0, false, Eigen::Vector3d::Zero());
				//bool twoContact2 = checkIfContatc(particles, param);
				//std::cout << "twoContact2 = " << twoContact << std::endl;
				
			}

	

			{
				std::ofstream outfile2(outputPath + "//bunny.obj", std::ios::trunc);
				for (int k = 0; k < particles.size(); k++)
				{
					Eigen::Vector3d scale = particles[k].position;
					outfile2 << std::scientific << std::setprecision(8) << "v " << scale[0] << " " << scale[1] << " " << scale[2] << std::endl;
				}
				outfile2.close();
			}


			std::cout << "particles.size() = " << particles.size() << std::endl;
			for (int i = 0; i <= 1000000; i++)
			{
				advanceStep(particles, param, partclieMaterial, i);
			}



		}




	}


	


	return 0;
};











