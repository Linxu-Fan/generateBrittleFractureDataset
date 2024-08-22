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

		std::random_device rd;
		int numBunnyInteriorPar = 20000;
		int numData = 10000;
		double sphereRadius = 0.02;
		int sphereParSkipRatio = 10;
		double sphereVelMag = 100;

		// simulation parameters
		parametersSim param;
		param.numOfThreads = numOfThreads;
		param.dt = 1.0E-6;
		param.dx = cbrt(2.0 / numBunnyInteriorPar) / 2;
		param.gravity = {0,0,0};
		param.updateDenpendecies();


		// initialize three types of materials
		Material material1;
		material1.density = 3000;
		material1.E = 3.2E12;
		material1.Gf = 3.2E5;
		material1.nu = 0.2;
		material1.bendingStressThreshold = 250;

		material1.thetaf = 8.0E9;
		double hsTarget = 0.45;
		material1.lch = hsTarget / (1 + hsTarget) / material1.calHsBar();
		material1.updateDenpendecies();
		std::cout << "material1.lch = " << material1.lch << std::endl;
		std::cout << "material1.HsBar * lch = " << material1.Hs << std::endl;

		std::vector<Material> partclieMaterial;
		partclieMaterial.push_back(material1);


		// Generate random seeds
		{
			//std::ofstream outfile4(outputPath + "//randomSeeds.txt", std::ios::trunc);
			//for (int h = 0; h < 100000; h++)
			//{

			//	std::random_device rd;
			//	std::mt19937 eng(rd());
			//	std::uniform_real_distribution<> distr(0.0, 1.0);

			//	double theta = 2 * PI * distr(eng); // Angle around the z-axis
			//	double phi = acos(1 - 2 * distr(eng)); // Angle from the z-axis

			//	double radius = 1.0; // Radius of the sphere

			//	// Convert spherical coordinates to Cartesian coordinates
			//	double x = radius * sin(phi) * cos(theta) + 0.5;
			//	double y = radius * sin(phi) * sin(theta) + 0.5;
			//	double z = radius * cos(phi) + 0.5;
			//	outfile4 << std::scientific << std::setprecision(8) << x << " " << y << " " << z << std::endl;

			//}
			//outfile4.close();
		}

		// Read seeds
		std::vector<Eigen::Vector3d> seeds;
		{
			std::ifstream in;
			in.open("./input/randomSeeds.txt");
			std::string line;

			while (getline(in, line))
			{
				if (line.size() > 0)
				{
					std::vector<std::string> vecCoor = split(line, " ");
					Eigen::Vector3d vt = { std::stod(vecCoor[0]) , std::stod(vecCoor[1]) , std::stod(vecCoor[2]) };
					seeds.push_back(vt);
				}
			}
			in.close();
		}

		
		objMesh bunnyMeshSurf, bunnyMeshInterior;
		readObjMesh("./input/bunny.obj", "./input/bunnyParticles_"+std::to_string(numBunnyInteriorPar) + ".obj", bunnyMeshSurf, bunnyMeshInterior, 0.95);
		objMesh sphereMeshSurfInterior;
		
		readSphereObjMesh("./input/sphereParticles.obj", sphereMeshSurfInterior, sphereRadius);
		int numBunnySurfVerts = bunnyMeshSurf.vertices.size();
		int numBunnyInteriorVerts = bunnyMeshInterior.vertices.size();

		// generate numData datasets	
		for (int k = 6743; k < numData; k++)
		{
			// Ste 1: read bunny particles
			std::vector<mpmParticle> particles;
			readBunnyMeshToParticles(particles, bunnyMeshInterior, 0, material1.density, param.dx * param.dx * param.dx / 8.0, true, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());


			// Step 2: read sphere particles
			int nearestPtInBunnyInterior = -99;
			{
				// Step 1: calcualte the diaplacement of shpere
				Eigen::Vector3d translation = seeds[k];
				std::vector<Eigen::Vector3d> vertSphere;
				for (int f = 0; f < sphereMeshSurfInterior.vertices.size(); f++)
				{
					if (f % sphereParSkipRatio == 0)
					{
						vertSphere.push_back(sphereMeshSurfInterior.vertices[f] + translation);
					}				
				}


				std::cout << "vertSphere.size() = " << vertSphere.size() << std::endl;

				std::ofstream outfilew3("./output/vtSphere.obj", std::ios::trunc);
				for (int kw = 0; kw < vertSphere.size(); kw++)
				{
					outfilew3 << std::scientific << std::setprecision(8) << "v " << vertSphere[kw][0] << " " << vertSphere[kw][1] << " " << vertSphere[kw][2] << std::endl;
				}
				outfilew3.close();



				// Step 2: calculate the nearest distance to the bunny surface
				double minDis = 1.0e8;
				int bunnyInteriorVt = -99, sphereInteriorVt = -99;
				for (int sv = 0; sv < vertSphere.size(); sv++)
				{
					for (int bv = 0; bv < bunnyMeshInterior.vertices.size(); bv++)
					{
						double dis = (vertSphere[sv] - bunnyMeshInterior.vertices[bv]).norm();
						if (dis < minDis)
						{
							minDis = dis;
							sphereInteriorVt = sv;
							bunnyInteriorVt = bv;						
						}
					}
				}
				nearestPtInBunnyInterior = bunnyInteriorVt;


				std::ofstream outfile12("./output/nearestPts.obj", std::ios::trunc);
				outfile12 << std::scientific << std::setprecision(8) << "v " << bunnyMeshInterior.vertices[bunnyInteriorVt][0] << " " << bunnyMeshInterior.vertices[bunnyInteriorVt][1] << " " << bunnyMeshInterior.vertices[bunnyInteriorVt][2] << " " << std::endl;
				outfile12 << std::scientific << std::setprecision(8) << "v " << vertSphere[sphereInteriorVt][0] << " " << vertSphere[sphereInteriorVt][1] << " " << vertSphere[sphereInteriorVt][2] << " " << std::endl;
				outfile12.close();

								
				// Step 3: move sphere particles to the target position
				std::cout << "Moving sphere particles to the target position!" << std::endl;
				Eigen::Vector3d dirToMoveOriginal = bunnyMeshInterior.vertices[bunnyInteriorVt] - vertSphere[sphereInteriorVt];
				Eigen::Vector3d dirToMove = dirToMoveOriginal;
				double res = 3.5 * param.dx * dirToMove.norm() / 1000;
				dirToMove = dirToMoveOriginal * (1.0 - 3.5 * param.dx / dirToMoveOriginal.norm());			
				std::cout << "res = " << res << std::endl;
				{					
					double minDisContact = -99.0;
					bool twoContact = false;
					std::cout << "d = ";
					for (int d = 0; d < 1000; d++)
					{
						std::cout << d << " , ";
						double ra = (double)d * res;
						dirToMove += dirToMove.normalized() * sphereRadius * ra;
						
						std::vector<Eigen::Vector3d> vertSphereFinalTmp;
						for (int f = 0; f < vertSphere.size(); f++)
						{
							vertSphereFinalTmp.push_back(vertSphere[f] + dirToMove);
						}

						objMesh sphereMeshParTmp;
						std::vector<mpmParticle> particlesTmp = particles;
						sphereMeshParTmp.vertices = vertSphereFinalTmp;
						readBunnyMeshToParticles(particlesTmp, sphereMeshParTmp, 0, material1.density, param.dx * param.dx * param.dx / 8.0, false, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
						twoContact = checkIfContatc(particlesTmp, param);
						if (twoContact)
						{
							minDisContact = ra - res;
							break;
						}
					}
					std::cout << std::endl;

					dirToMove += dirToMove.normalized() * sphereRadius * minDisContact;
					std::vector<Eigen::Vector3d> vertSphereFinal;
					for (int f = 0; f < vertSphere.size(); f++)
					{
						vertSphereFinal.push_back(vertSphere[f] + dirToMove);
					}
					sphereMeshSurfInterior.vertices = vertSphereFinal;
					readBunnyMeshToParticles(particles, sphereMeshSurfInterior, 0, material1.density, param.dx * param.dx * param.dx / 8.0, false, Eigen::Vector3d::Zero(), dirToMove.normalized() * sphereVelMag);
					bool twoContact2 = checkIfContatc(particles, param);
				}
						


				param.force_position = bunnyMeshInterior.vertices[bunnyInteriorVt];
				param.force_direction = dirToMove.normalized();
				param.force_magnitude = 200;
				
			}

	
			// Step 3: find nearnest point in the bunny's surface
			{
				int nearestPtInBunnySurf = -99;
				double minDisIS = 1.0E8;
				std::cout << "nearestPtInBunnyInterior = " << nearestPtInBunnyInterior << std::endl;
				Eigen::Vector3d interiorPtCoor = bunnyMeshInterior.vertices[nearestPtInBunnyInterior];
				for (int g = 0; g < bunnyMeshSurf.vertices.size(); g++)
				{
					if ((bunnyMeshSurf.vertices[g] - interiorPtCoor).norm() < minDisIS)
					{
						minDisIS = (bunnyMeshSurf.vertices[g] - interiorPtCoor).norm();
						nearestPtInBunnySurf = g;
					}
				}
			}



			std::cout << "particles.size() = " << particles.size() << std::endl;
			for (int timestep = 0; timestep <= 1000000; timestep++)
			{
				std::cout << "timestep = " << timestep << std::endl;
				if(timestep % 10 == 0)
				{
					std::ofstream outfile2(outputPath + "//bunny_"+std::to_string(timestep) + ".obj", std::ios::trunc);
					std::ofstream outfile4(outputPath + "//bunnyStress_"+std::to_string(timestep) + ".txt", std::ios::trunc);
					for (int k = 0; k < numBunnyInteriorVerts; k++)
					{
						Eigen::Vector3d scale = particles[k].position;

						Eigen::Matrix3d F = particles[k].F;
						double J = particles[k].F.determinant();
						int materialIndex = particles[k].material;
						Eigen::Matrix3d cauchyStressE = (partclieMaterial[materialIndex].lambda * log(J) / J - partclieMaterial[materialIndex].mu / J) * Eigen::Matrix3d::Identity() + partclieMaterial[materialIndex].mu / J * F * F.transpose();
						Eigen::EigenSolver<Eigen::MatrixXd> es(cauchyStressE);
						Eigen::Vector3d eigenValues = { es.eigenvalues()[0].real(), es.eigenvalues()[1].real(), es.eigenvalues()[2].real() };
						Eigen::Matrix3d eigenVectors;
						eigenVectors << es.eigenvectors().real();
						double maxEigenValue = std::max(std::max(eigenValues[0], eigenValues[1]), eigenValues[2]);


						outfile2 << std::scientific << std::setprecision(8) << "v " << scale[0] << " " << scale[1] << " " << scale[2] << " " << maxEigenValue << std::endl;
						outfile4 << std::scientific << std::setprecision(8) << scale[0] << " " << scale[1] << " " << scale[2] << " " << maxEigenValue << " " << particles[k].dp << std::endl;
					}
					outfile2.close();
					outfile4.close();

					std::ofstream outfile3(outputPath + "//sphere_" + std::to_string(timestep) + ".obj", std::ios::trunc);
					for (int k = numBunnyInteriorVerts; k < particles.size(); k++)
					{
						Eigen::Vector3d scale = particles[k].position;

						Eigen::Matrix3d F = particles[k].F;
						double J = particles[k].F.determinant();
						int materialIndex = particles[k].material;
						Eigen::Matrix3d cauchyStressE = (partclieMaterial[materialIndex].lambda * log(J) / J - partclieMaterial[materialIndex].mu / J) * Eigen::Matrix3d::Identity() + partclieMaterial[materialIndex].mu / J * F * F.transpose();
						Eigen::EigenSolver<Eigen::MatrixXd> es(cauchyStressE);
						Eigen::Vector3d eigenValues = { es.eigenvalues()[0].real(), es.eigenvalues()[1].real(), es.eigenvalues()[2].real() };
						Eigen::Matrix3d eigenVectors;
						eigenVectors << es.eigenvectors().real();
						double maxEigenValue = std::max(std::max(eigenValues[0], eigenValues[1]), eigenValues[2]);

						outfile3 << std::scientific << std::setprecision(8) << "v " << scale[0] << " " << scale[1] << " " << scale[2] << " " << maxEigenValue << std::endl;
					}
					outfile3.close();
				}

				advanceStep(particles, param, partclieMaterial, timestep);
			}





		}






	}


	


	return 0;
};











