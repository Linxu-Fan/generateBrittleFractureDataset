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
		param.dt = 2.0E-3;
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
		// Add a cylinder composed of particles
		double radiusGround = 0.3;
		for (int xi = -round(radiusGround / (param.dx / 2.0)); xi <= round(radiusGround / (param.dx / 2.0)); xi++)
		{
			for (int yi = -round(radiusGround / (param.dx / 2.0)); yi <= round(radiusGround / (param.dx / 2.0)); yi++)
			{
				for (int zi = -round(radiusGround / (param.dx / 2.0)); zi <= round(radiusGround / (param.dx / 2.0)); zi++)
				{
					Eigen::Vector3d coor = { xi * param.dx / 2.0 , yi * param.dx / 2.0, zi * param.dx / 2.0 };
					if (coor.norm() <= radiusGround)
					{
						// sphere 1
						//Eigen::Vector3d devia1 = { -10.124,0.124,0.124 };
						Eigen::Vector3d devia1 = { 1.0,1.0,1.0 };
						{
							mpmParticle par1;
							par1.position = coor + devia1;
							par1.velocity = { 0,0,0 };
							par1.volume = param.dx * param.dx * param.dx;
							par1.material = 0;
							par1.mass = partclieMaterial[par1.material].density * par1.volume;
							par1.color = 1; // we set these particles to be still as we don't want them to move
							particles.push_back(par1);
						}
						//// sphere 1
						////Eigen::Vector3d devia2 = { -9.124,0.124,0.124 };
						//Eigen::Vector3d devia2 = { 2.0,1.0,1.0 };
						//{
						//	mpmParticle par1;
						//	par1.position = coor + devia2;
						//	par1.velocity = { -1.0,0,0 };
						//	par1.volume = param.dx * param.dx * param.dx;
						//	par1.material = 0;
						//	par1.mass = partclieMaterial[par1.material].density * par1.volume;
						//	par1.color = 0; // we set these particles to be still as we don't want them to move
						//	particles.push_back(par1);
						//}
					}

				}
			}
		}


		{
			std::ofstream outfile2(outputPath + "//sphere.obj", std::ios::trunc);
			for (int k = 0; k < particles.size(); k++)
			{
				Eigen::Vector3d scale = particles[k].position;
				outfile2 << std::scientific << std::setprecision(8) << "v " << scale[0] << " " << scale[1] << " " << scale[2] << std::endl;
			}
			outfile2.close();
		}
		
		int spSize = particles.size();

		{
			double radiusGround = 0.5;
			double dx = 0.01;
			for (int yi = 0; yi <= 100; yi++)
			{
				for (int xi = 0; xi <= 100; xi++)
				{
					double xc = (double)xi * dx + 0.5, yc = (double)yi * dx + 0.5;
					mpmParticle par1;
					par1.position = {xc, yc, 1.6};
					par1.velocity = { 0,0,-1.0 };
					par1.volume = param.dx * param.dx * param.dx;
					par1.material = 0;
					par1.mass = partclieMaterial[par1.material].density * par1.volume;
					par1.color = 0; // we set these particles to be still as we don't want them to move
					particles.push_back(par1);
				}
			}

			
		}



		std::cout << "particles.size() = " << particles.size() << std::endl;
		for (int i = 0; i <= 1000000; i++)
		{
			if (i % 50 == 0)
			{
				std::cout << "timestep = " << i << std::endl;
				std::ofstream outfile2("./output/squaredCloth_"+std::to_string(i) + ".obj", std::ios::trunc);
				for (int k = spSize; k < particles.size(); k++)
				{
					Eigen::Vector3d scale = particles[k].position;
					outfile2 << std::scientific << std::setprecision(8) << "v " << scale[0] << " " << scale[1] << " " << scale[2] << std::endl;
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




				//std::cout << "Step = " << i << std::endl;
				//std::ofstream outfile2(outputPath + "//objects_" + std::to_string(i) + ".obj", std::ios::trunc);
				//for (int k = 0; k < particles.size(); k++)
				//{
				//	Eigen::Vector3d scale = particles[k].position;
				//	outfile2 << std::scientific << std::setprecision(8) << "v " << scale[0] << " " << scale[1] << " " << scale[2] << std::endl;
				//}
				//outfile2.close();
			}

			advanceStep(particles, param, partclieMaterial, i);
		}


	}


	


	return 0;
};











