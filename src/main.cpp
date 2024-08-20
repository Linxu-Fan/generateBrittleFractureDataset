#include "weights.h"
#include "particles.h"
#include "grid.h"
#include "advance.h"



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
		param.dx = 0.04;
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


		std::vector<mpmParticle> particles;
		objMesh bunnyMeshSurf, bunnyMeshPars;
		readObjMesh("./input/bunny.obj", "./input/bunnyParticles.obj", bunnyMeshSurf, bunnyMeshPars, 0.95);
		readBunnyMeshToParticles(particles, bunnyMeshPars, 0, material1.density, param.dx * param.dx * param.dx / 8.0, true, Eigen::Vector3d::Zero(), 1);

		objMesh sphereMeshPar;
		readSphereObjMesh("./input/sphereParticles.obj", sphereMeshPar, 0.02);
		readBunnyMeshToParticles(particles, sphereMeshPar, 0, material1.density, param.dx * param.dx * param.dx / 8.0, false, Eigen::Vector3d::Zero(), 10);


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
			if (i % 50 == 0)
			{
			
			}

			advanceStep(particles, param, partclieMaterial, i);
		}


	}


	


	return 0;
};











