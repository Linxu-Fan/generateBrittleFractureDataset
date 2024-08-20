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


// read .msh mesh file
void Mesh::readMesh(std::string filePath, double thicknessMesh)
{
	thickness = thicknessMesh;

	// read obj file
	std::ifstream in;
	in.open(filePath);
	std::string line;
	while (getline(in, line))
	{
		if (line.size() > 0)
		{
			std::vector<std::string> vecCoor = split(line, " ");
			if (vecCoor[0] == "v")
			{
				Eigen::Vector3d pos = { std::stod(vecCoor[1]) , std::stod(vecCoor[2]) , std::stod(vecCoor[3]) };
				pos_node.push_back(pos);
				vel_node.push_back(Eigen::Vector3d::Zero());
			}
			if (vecCoor[0] == "f")
			{
				Eigen::Vector3i tri = { std::stoi(vecCoor[1]) - 1 , std::stoi(vecCoor[2]) - 1 , std::stoi(vecCoor[3]) - 1 };
				triangles.push_back(tri);
			}			
		}
	}
	in.close();


	// calculate the area and volume of each node
	{
		vol_node.resize(pos_node.size());
		for (int i = 0; i < triangles.size(); i++)
		{
			Eigen::Vector3i tri = triangles[i];
			int v0 = tri[0], v1 = tri[1], v2 = tri[2];
			Eigen::Vector3d p0 = pos_node[v0], p1 = pos_node[v1], p2 = pos_node[v2];
			double area = calculateTriangleArea3D(p0, p1, p2);

			area_triangle.push_back(area);

			double vol = area * thicknessMesh;
			vol_node[v0] += vol / 3.0;
			vol_node[v1] += vol / 3.0;
			vol_node[v2] += vol / 3.0;
		}
	}


	// find triangles that share each vertex
	{
		node_triangles.resize(pos_node.size());
		for (int i = 0; i < triangles.size(); i++)
		{
			Eigen::Vector3i tri = triangles[i];
			int v0 = tri[0], v1 = tri[1], v2 = tri[2];
			node_triangles[v0].push_back(i);
			node_triangles[v1].push_back(i);
			node_triangles[v2].push_back(i);
		}
	}


	// find the boundary node
	{

		edgeTris.clear();
		for (int i = 0; i < triangles.size(); i++)
		{
			Eigen::Vector3i tri = triangles[i];
			int v0 = tri[0], v1 = tri[1], v2 = tri[2];

			std::string e0 = std::to_string(std::min(v0, v1)) + "#" + std::to_string(std::max(v0, v1));
			std::string e1 = std::to_string(std::min(v1, v2)) + "#" + std::to_string(std::max(v1, v2));
			std::string e2 = std::to_string(std::min(v2, v0)) + "#" + std::to_string(std::max(v2, v0));

			if (edgeTris.find(e0) == edgeTris.end())
			{
				edgeTris[e0][0] = i;
				edgeTris[e0][1] = -99;
			}
			else
			{
				edgeTris[e0][1] = i;
			}

			if (edgeTris.find(e1) == edgeTris.end())
			{
				edgeTris[e1][0] = i;
				edgeTris[e1][1] = -99;
			}
			else
			{
				edgeTris[e1][1] = i;
			}

			if (edgeTris.find(e2) == edgeTris.end())
			{
				edgeTris[e2][0] = i;
				edgeTris[e2][1] = -99;
			}
			else
			{
				edgeTris[e2][1] = i;
			}

		}

		for (std::map<std::string, Eigen::Vector2i>::iterator it = edgeTris.begin(); it != edgeTris.end(); it++)
		{
			if (it->second[1] == -99)
			{
				std::string name = it->first;
				std::vector<std::string> vecCoor = split(name, "#");
				ifBoundary_node[std::stoi(vecCoor[0])] = true;
				ifBoundary_node[std::stoi(vecCoor[1])] = true;
			}
		}

	}



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


void Mesh::calUndeformedFundamentalForm()
{
	// 1. calculate the triangle's normal
	for (int i = 0; i < triangles.size(); i++)
	{
		Eigen::Vector3i tri = triangles[i];
		int v0 = tri[0], v1 = tri[1], v2 = tri[2];

		Eigen::Vector3d e1 = pos_node[v0] - pos_node[v1];
		Eigen::Vector3d e2 = pos_node[v2] - pos_node[v1];

		Eigen::Vector3d normal = e1.cross(e2).normalized();

		normal_triangle.push_back(normal);
	}

	// 2. calculate the triangle's mid-edge normal
	for (int i = 0; i < triangles.size(); i++)
	{
		Eigen::Vector3i tri = triangles[i];
		int v0 = tri[0], v1 = tri[1], v2 = tri[2];

		std::string e0 = std::to_string(std::min(v0, v1)) + "#" + std::to_string(std::max(v0, v1));
		std::string e1 = std::to_string(std::min(v1, v2)) + "#" + std::to_string(std::max(v1, v2));
		std::string e2 = std::to_string(std::min(v2, v0)) + "#" + std::to_string(std::max(v2, v0));


		Eigen::Matrix3d midEdgeNorm = Eigen::Matrix3d::Zero();

		Eigen::Vector2i tris = edgeTris[e0];
		if (tris[1] = -99)
		{
			midEdgeNorm.col(2) = normal_triangle[i];
		}
		else
		{
			midEdgeNorm.col(2) = 0.5 * (normal_triangle[tris[0]] + normal_triangle[tris[1]]);
		}

		tris = edgeTris[e1];
		if (tris[1] = -99)
		{
			midEdgeNorm.col(0) = normal_triangle[i];
		}
		else
		{
			midEdgeNorm.col(0) = 0.5 * (normal_triangle[tris[0]] + normal_triangle[tris[1]]);
		}

		tris = edgeTris[e2];
		if (tris[1] = -99)
		{
			midEdgeNorm.col(1) = normal_triangle[i];
		}
		else
		{
			midEdgeNorm.col(1) = 0.5 * (normal_triangle[tris[0]] + normal_triangle[tris[1]]);
		}

		midEdge_normal_triangle.push_back(midEdgeNorm);
	}

	// 3. calculate T, Q, a_bar, b_bar
	for (int i = 0; i < triangles.size(); i++)
	{
		Eigen::Vector3i tri = triangles[i];
		int vi = tri[0], vj = tri[1], vk = tri[2];

		Eigen::Vector3d Xi = pos_node[vi], Xj = pos_node[vj], Xk = pos_node[vk];
		Eigen::Vector3d Ni = midEdge_normal_triangle[i].col(0), Nj = midEdge_normal_triangle[i].col(1), Nk = midEdge_normal_triangle[i].col(2);
		Eigen::Vector3d Nijk = normal_triangle[i];


		Eigen::Matrix3d T = Eigen::Matrix3d::Zero(), Q = Eigen::Matrix3d::Zero();
		Eigen::Matrix2d a_bar = Eigen::Matrix2d::Zero(), b_bar = Eigen::Matrix2d::Zero();


		T.col(0) = Xj - Xi;
		T.col(1) = Xk - Xi;
		T.col(2) = Nijk;

		Q.col(0) = 2.0 * (Ni - Nj);
		Q.col(1) = 2.0 * (Ni - Nk);
	
		a_bar(0, 0) = (Xj - Xi).dot(Xj - Xi);
		a_bar(0, 1) = (Xj - Xi).dot(Xk - Xi);
		a_bar(1, 0) = (Xj - Xi).dot(Xk - Xi);
		a_bar(1, 1) = (Xk - Xi).dot(Xk - Xi);

		b_bar(0, 0) = (Ni - Nj).dot(Xi - Xj);
		b_bar(0, 1) = (Ni - Nj).dot(Xi - Xk);
		b_bar(1, 0) = (Ni - Nk).dot(Xi - Xj);
		b_bar(1, 1) = (Ni - Nk).dot(Xi - Xk);

		// undeformed curvature and determinant
		Eigen::Matrix2d l_bar = a_bar.inverse() * b_bar;
		double H = (l_bar).trace() / 2.0;
		double K = (l_bar).determinant();

		T_triangle.push_back(T);
		Q_triangle.push_back(Q);
		l_bar_triangle.push_back(l_bar);
		H_triangle.push_back(H);
		K_triangle.push_back(K);


	}


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



