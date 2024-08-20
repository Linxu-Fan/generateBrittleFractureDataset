#include "particles.h"

// initialize MPM particles from mesh
void readBunnyMeshToParticles(std::vector<mpmParticle>& particles, objMesh& bunnyMesh, int materialIndex, double density, double voume, bool breakable, Eigen::Vector3d translation, int skipRatio)
{
	for (int i = 0; i < bunnyMesh.vertices.size(); i++)
	{
		if (i % skipRatio == 0)
		{
			mpmParticle par1;
			par1.position = bunnyMesh.vertices[i] + translation;
			par1.velocity = Eigen::Vector3d::Zero();
			par1.volume = voume;
			par1.material = materialIndex;
			par1.mass = density * par1.volume;
			par1.color = 0;
			par1.breakable = breakable;
			particles.push_back(par1);
		}

	}
}


void readObjMesh(std::string meshSurfPath, std::string meshParPath, objMesh& meshSurf, objMesh& meshPars, double size)
{
	std::ifstream in;
	in.open(meshSurfPath);
	std::string line;

	while (getline(in, line))
	{
		if (line.size() > 0)
		{
			std::vector<std::string> vecCoor = split(line, " ");
			if (vecCoor[0] == "v")
			{
				Eigen::Vector3d vt = {std::stod(vecCoor[1]) , std::stod(vecCoor[2]) , std::stod(vecCoor[3]) };
				meshSurf.vertices.push_back(vt);
			}

			if (vecCoor[0] == "f")
			{
				Eigen::Vector3i fc = { std::stoi(vecCoor[1]) - 1 , std::stoi(vecCoor[2]) - 1 , std::stoi(vecCoor[3]) - 1 };
				meshSurf.faces.push_back(fc);
			}
		}
	}
	in.close();

	meshSurf.calMinMaxCoor();
	meshSurf.resizeAndRemove(size);



	in.open(meshParPath);
	while (getline(in, line))
	{
		if (line.size() > 0)
		{
			std::vector<std::string> vecCoor = split(line, " ");
			if (vecCoor[0] == "v")
			{
				Eigen::Vector3d vt = { std::stod(vecCoor[1]) , std::stod(vecCoor[2]) , std::stod(vecCoor[3]) };
				meshPars.vertices.push_back(vt);
			}
		}
	}
	in.close();
	meshPars.minMaxCoor = meshSurf.minMaxCoor;
	meshPars.resizeAndRemove(size);

}



void readSphereObjMesh(std::string meshSpherePath, objMesh& meshSphere, double size)
{
	std::ifstream in;
	in.open(meshSpherePath);
	std::string line;

	while (getline(in, line))
	{
		if (line.size() > 0)
		{
			std::vector<std::string> vecCoor = split(line, " ");
			if (vecCoor[0] == "v")
			{
				Eigen::Vector3d vt = { std::stod(vecCoor[1]) * size , std::stod(vecCoor[2]) * size , std::stod(vecCoor[3]) * size };
				meshSphere.vertices.push_back(vt);
			}
		}
	}
	in.close();

}
