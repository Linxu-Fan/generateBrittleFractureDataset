#include "weights.h"
#include "particles.h"
#include "grid.h"
#include "advance.h"
#include "utils.h"



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
		int numOfThreads = 40;
		std::string outputPath = ".//output";

			
		#ifdef _WIN32
			if (mkdir(".//output") == -1)
			{
				std::cout << " Error : " << strerror(errno) << std::endl;
			}
		#else
			if (mkdir(".//output", 0777) == -1)
			{
				std::cout << " Error : " << strerror(errno) << std::endl;
			}
		#endif


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
		param.gravity = { 0,0,0 };
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


		objMesh bunnyMeshSurf_org, bunnyMeshInterior_org;
		readObjMesh("../input/lion.obj", "../input/lionParticles_" + std::to_string(numBunnyInteriorPar) + ".obj", bunnyMeshSurf_org, bunnyMeshInterior_org, 0.95);
		objMesh sphereMeshSurfInterior_org;

		readSphereObjMesh("../input/sphereParticles.obj", sphereMeshSurfInterior_org, sphereRadius);
		int numBunnySurfVerts = bunnyMeshSurf_org.vertices.size();
		int numBunnyInteriorVerts = bunnyMeshInterior_org.vertices.size();

		// calculate the bunny's COM
		Eigen::Vector3d COM = { 0,0,0 };
		for (int bv = 0; bv < bunnyMeshInterior_org.vertices.size(); bv++)
		{
			COM += bunnyMeshInterior_org.vertices[bv];
		}
		COM = COM / (double)bunnyMeshInterior_org.vertices.size();



		{
			std::ofstream outfile2("./output/lion_mesh.obj", std::ios::trunc);
			for (int vert = 0; vert < bunnyMeshSurf_org.vertices.size(); ++vert)
			{
				outfile2 << std::scientific << std::setprecision(8) << "v " << bunnyMeshSurf_org.vertices[vert][0] << " " << bunnyMeshSurf_org.vertices[vert][1] << " " << bunnyMeshSurf_org.vertices[vert][2] << " " << std::endl;
			}
			for (int face = 0; face < bunnyMeshSurf_org.faces.size(); ++face)
			{
				outfile2 << std::scientific << std::setprecision(8) << "f ";
				for (int vert = 0; vert < bunnyMeshSurf_org.faces[face].size(); ++vert) {
					outfile2 << std::scientific << std::setprecision(8) << bunnyMeshSurf_org.faces[face][vert] + 1 << " ";
				}
				outfile2 << std::endl;
			}
			outfile2.close();
		}



		// generate numData datasets	
		for (int k = 329; k < 700; k++)
		{
			int surfParIndex = k * 70, interiorParIndex = -99;;
			std::vector<Eigen::Vector3d> vertSphere;
			for (int f = 0; f < sphereMeshSurfInterior_org.vertices.size(); f++)
			{
				if (f % sphereParSkipRatio == 0)
				{
					vertSphere.push_back(sphereMeshSurfInterior_org.vertices[f] - Eigen::Vector3d::Ones());
				}
			}


			// Ste 1: read bunny particles
			std::vector<mpmParticle> particles;
			readBunnyMeshToParticles(particles, bunnyMeshInterior_org, 0, material1.density, param.dx * param.dx * param.dx / 8.0, true, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());


			// Step 2: find nearest interior particles inside the bunny
			double minDis = 1.0e8;
			for (int bv = 0; bv < bunnyMeshInterior_org.vertices.size(); bv++)
			{
				double dis = (bunnyMeshSurf_org.vertices[surfParIndex] - bunnyMeshInterior_org.vertices[bv]).norm();
				if (dis < minDis)
				{
					minDis = dis;
					interiorParIndex = bv;
				}
			}
			param.force_positionParIndex = interiorParIndex;
			param.force_direction = (COM - bunnyMeshInterior_org.vertices[interiorParIndex]).normalized();
			param.force_magnitude = 400;


			// Step 3: calculate the contact normal
			Eigen::Vector3d triNormalVert = { 0,0,0 };
			{
				for (int p = 0; p < bunnyMeshSurf_org.vertTris[surfParIndex].size(); p++)
				{
					int triIndex = bunnyMeshSurf_org.vertTris[surfParIndex][p];
					Eigen::Vector3i triVts = bunnyMeshSurf_org.faces[triIndex];

					Eigen::Vector3d a = bunnyMeshSurf_org.vertices[triVts[0]];
					Eigen::Vector3d b = bunnyMeshSurf_org.vertices[triVts[1]];
					Eigen::Vector3d c = bunnyMeshSurf_org.vertices[triVts[2]];

					triNormalVert += (b - a).cross(c - a).normalized();
				}
				triNormalVert = triNormalVert.normalized();
			}


	
			// Step 4: read sphere particles
			std::vector<Eigen::Vector3d> vertSphereFinal;
			for (int f = 0; f < vertSphere.size(); f++)
			{
				vertSphereFinal.push_back(vertSphere[f] + triNormalVert.normalized() * sphereRadius * 1.5 + bunnyMeshInterior_org.vertices[interiorParIndex]);
			}
			objMesh sphereMeshSurfInterior = sphereMeshSurfInterior_org;
			sphereMeshSurfInterior.vertices = vertSphereFinal;
			readBunnyMeshToParticles(particles, sphereMeshSurfInterior, 0, material1.density, param.dx * param.dx * param.dx / 8.0, false, Eigen::Vector3d::Zero(), param.force_direction * sphereVelMag);




			bool generateFragments = false;
			bool simulationFailed = false;
			std::tuple<bool, meshObjFormat, meshObjFormat, std::vector<meshObjFormat>> crackSurfs;
			for (int timestep = 0; timestep <= 2000 && generateFragments == false && !simulationFailed; timestep++)
			{
				std::cout <<"Trial = "<< k <<"; timestep = " << timestep << std::endl;



				advanceStep(particles, param, partclieMaterial, timestep);

				// The simulation fails due to unexpected reasons. Test
				for (int k = 0; k < numBunnyInteriorVerts; k++)
				{
					Eigen::Vector3d scale = particles[k].position;
					if (scale.hasNaN() || scale.array().isInf().any())
					{
						std::cout << "Conataining inf or NaN, exit!" << std::endl << std::endl << std::endl;
						simulationFailed = true;
						break;
					}
				}

				if (timestep % 20 == 0 && !simulationFailed)
				{
					/*{
						std::ofstream outfile2(outputPath + "//particles_"+std::to_string(k)+"_"+std::to_string(timestep)+".txt", std::ios::trunc);
						for (int k = 0; k < numBunnyInteriorVerts; k++)
						{
							Eigen::Vector3d scale = particles[k].position;

							outfile2 << std::scientific << std::setprecision(8) << "v " << scale[0] << " " << scale[1] << " " << scale[2] << std::endl;
						}
						outfile2.close();

					}*/


					crackSurfs = tryToExtractCracks(particles, param, timestep);
					if (std::get<0>(crackSurfs) == true && std::get<3>(crackSurfs).size() > 1)
					{
						generateFragments = true;
					}
				}
				
			}
			
			if (generateFragments == true)
			{

				std::string folder = ".//output//VT_" + std::to_string(k);



			
				#ifdef _WIN32
					if (mkdir(folder.c_str()) == -1)
					{
						std::cout << " Error : " << strerror(errno) << std::endl;
					}
				#else
					if (mkdir(folder.c_str(), 0777) == -1)
					{
						std::cout << " Error : " << strerror(errno) << std::endl;
					}
				#endif


				// output crack surface and fragments
				writeObjFile_fullName(std::get<2>(crackSurfs).vertices, std::get<2>(crackSurfs).faces, folder + "/crackSurfaceFull.obj", true);
				for (int fag = 0; fag < std::get<3>(crackSurfs).size(); fag++)
				{
					writeObjFile_fullName(std::get<3>(crackSurfs)[fag].vertices, std::get<3>(crackSurfs)[fag].faces, folder + "/fragment_"+std::to_string(fag) + ".obj", true);
				}


				{
					std::ofstream outfile2(outputPath + "//lion.obj", std::ios::trunc);
					for (int k = 0; k < numBunnyInteriorVerts; k++)
					{
						Eigen::Vector3d scale = particles[k].position;

						outfile2 << std::scientific << std::setprecision(8) << "v " << scale[0] << " " << scale[1] << " " << scale[2] << std::endl;
					}
					outfile2.close();

				}

				// output contact force
				{
					std::ofstream outfile4(folder + "/contact.txt", std::ios::trunc);
					Eigen::Vector3d ptSphere = bunnyMeshSurf_org.vertices[surfParIndex];
					Eigen::Vector3d ptBunny = param.force_direction ;
					outfile4 << std::scientific << std::setprecision(8) << "Force point: " << ptSphere[0] << " " << ptSphere[1] << " " << ptSphere[2] << std::endl;
					outfile4 << std::scientific << std::setprecision(8) << "Force direction:  " << ptBunny[0] << " " << ptBunny[1] << " " << ptBunny[2] << std::endl;
					outfile4.close();
				}


			}
			else
			{
				std::cout << "Failed to generate cracks. Skip this trial!" << std::endl;
			}



		}




	}





	return 0;
};













