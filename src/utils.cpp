#include "utils.h"



int calculateID(int x, int y, int z, Eigen::Vector3d len, double dx) // coordinate of x and y, length in three dimensions of the cube, grid space
{
	Eigen::Vector3i length = (len / dx).cast<int>() + Eigen::Vector3i::Constant(1);
	int ID = z * (length(0) * length(1)) + y * length(0) + x;
	return ID;
};


std::string calculateID_string(int x, int y, int z) // string id of the particle
{
	return std::to_string(x) + "#" + std::to_string(y) + "#" + std::to_string(z);
};


// split a line from a text file
std::vector<std::string> split(const std::string& s, const std::string& seperator) {
	std::vector<std::string> result;
	typedef std::string::size_type string_size;
	string_size i = 0;

	while (i != s.size()) {
		int flag = 0;
		while (i != s.size() && flag == 0) {
			flag = 1;
			for (string_size x = 0; x < seperator.size(); ++x)
				if (s[i] == seperator[x]) {
					++i;
					flag = 0;
					break;
				}
		}

		flag = 0;
		string_size j = i;
		while (j != s.size() && flag == 0) {
			for (string_size x = 0; x < seperator.size(); ++x)
				if (s[j] == seperator[x]) {
					flag = 1;
					break;
				}
			if (flag == 0)
				++j;
		}
		if (i != j) {
			result.push_back(s.substr(i, j - i));
			i = j;
		}
	}
	return result;
}


// Given the vertices and faces information, write the obj file.
void writeObjFile(std::vector<Eigen::Vector3d> vertices, std::vector<std::vector<int>> faces, std::string name, bool startFrom0 = true)
{
	std::ofstream outfile2("./output/" + name + ".obj", std::ios::trunc);
	for (int vert = 0; vert < vertices.size(); ++vert)
	{
		outfile2 << std::scientific << std::setprecision(8) << "v " << vertices[vert][0] << " " << vertices[vert][1] << " " << vertices[vert][2] << " " << std::endl;
	}

	if (startFrom0)
	{
		for (int face = 0; face < faces.size(); ++face)
		{
			outfile2 << std::scientific << std::setprecision(8) << "f ";
			for (int vert = 0; vert < faces[face].size(); ++vert) {
				outfile2 << std::scientific << std::setprecision(8) << faces[face][vert] + 1 << " ";
			}
			outfile2 << std::endl;
		}
	}
	else
	{
		for (int face = 0; face < faces.size(); ++face)
		{
			outfile2 << std::scientific << std::setprecision(8) << "f ";
			for (int vert = 0; vert < faces[face].size(); ++vert) {
				outfile2 << std::scientific << std::setprecision(8) << faces[face][vert] << " ";
			}
			outfile2 << std::endl;
		}
	}


	outfile2.close();

}


// Given the vertices information, write an obj file and an txt file.
void writeObjFile(std::vector<Eigen::Vector3d> vertices, std::string name)
{
	std::ofstream outfile2("./output/" + name + ".obj", std::ios::trunc);
	std::ofstream outfile3("./output/" + name + ".txt", std::ios::trunc);
	for (int vert = 0; vert < vertices.size(); ++vert)
	{
		outfile2 << std::scientific << std::setprecision(8) << "v " << vertices[vert][0] << " " << vertices[vert][1] << " " << vertices[vert][2] << " " << std::endl;
		outfile3 << std::scientific << std::setprecision(8) << vertices[vert][0] << " " << vertices[vert][1] << " " << vertices[vert][2] << " " << std::endl;
	}
	outfile2.close();
	outfile3.close();

}


double calculateTriangleArea3D(Eigen::Vector3d p0, Eigen::Vector3d p1, Eigen::Vector3d p2)
{
	// Calculate vectors v1 and v2
	Eigen::Vector3d vec1 = p1 - p0;
	Eigen::Vector3d vec2 = p2 - p0;

	// Compute the cross product of vec1 and vec2
	Eigen::Vector3d crossProduct = vec1.cross(vec2);

	// Compute the magnitude of the cross product
	double crossMagnitude = crossProduct.norm();

	// The area of the triangle is half the magnitude of the cross product
	return 0.5 * crossMagnitude;
}




Eigen::Matrix3d getSkewMatrix(Eigen::Vector3d vc)
{
	Eigen::Matrix3d res = Eigen::Matrix3d::Zero();
	res(0, 1) = -vc[2];
	res(0, 2) = vc[1];
	res(1, 0) = vc[2];
	res(1, 2) = -vc[0];
	res(2, 0) = -vc[1];
	res(2, 1) = vc[0];
	return res;
}


std::map<int, std::map<int, Eigen::Matrix3d>> calMatrixProductDerivative(Eigen::Matrix3d& B)
{
	std::map<int, std::map<int, Eigen::Matrix3d>> res;

	Eigen::Matrix3d zm = Eigen::Matrix3d::Zero();
	
	// 0,0
	zm = Eigen::Matrix3d::Zero();
	zm.row(0) = B.col(0);
	res[0][0] = zm;

	// 1,0
	zm = Eigen::Matrix3d::Zero();
	zm.row(1) = B.col(0);
	res[1][0] = zm;

	// 2,0
	zm = Eigen::Matrix3d::Zero();
	zm.row(2) = B.col(0);
	res[2][0] = zm;

	// 0,1
	zm = Eigen::Matrix3d::Zero();
	zm.row(0) = B.col(1);
	res[0][1] = zm;

	// 1,1
	zm = Eigen::Matrix3d::Zero();
	zm.row(1) = B.col(1);
	res[1][1] = zm;

	// 2,1
	zm = Eigen::Matrix3d::Zero();
	zm.row(2) = B.col(1);
	res[2][1] = zm;

	// 0,2
	zm = Eigen::Matrix3d::Zero();
	zm.row(0) = B.col(2);
	res[0][2] = zm;

	// 1,2
	zm = Eigen::Matrix3d::Zero();
	zm.row(1) = B.col(2);
	res[1][2] = zm;

	// 2,2
	zm = Eigen::Matrix3d::Zero();
	zm.row(2) = B.col(2);
	res[2][2] = zm;


	return res;
}


void objMesh::calMinMaxCoor()
{
	double maxx = -1.0E9, maxy = -1.0E9, maxz = -1.0E9;
	double minx = 1.0E9, miny = 1.0E9, minz = 1.0E9;
	for (auto ver : vertices)
	{
		maxx = std::max(maxx, ver.x());
		maxy = std::max(maxy, ver.y());
		maxz = std::max(maxz, ver.z());

		minx = std::min(minx, ver.x());
		miny = std::min(miny, ver.y());
		minz = std::min(minz, ver.z());
	}
	Eigen::Vector3d min = { minx, miny, minz };
	Eigen::Vector3d max = { maxx, maxy, maxz };
	minMaxCoor.first = min;
	minMaxCoor.second = max;
}


void objMesh::resizeAndRemove(double size)
{
	Eigen::Vector3d min = minMaxCoor.first;
	Eigen::Vector3d max = minMaxCoor.second;

	double maxDim = std::max(max[0] - min[0], std::max(max[1] - min[1], max[2] - min[2]));
	double ratio = size / maxDim;
	std::cout << "ratio = " << ratio << std::endl;
	for (int i = 0; i < vertices.size(); i++)
	{
		Eigen::Vector3d move = { (1 - size) / 2,(1 - size) / 2,(1 - size) / 2 };
		Eigen::Vector3d vert = vertices[i] - min + move;
		vert = (vert - move) * ratio + move;
		vertices[i] = vert;
	}

}

