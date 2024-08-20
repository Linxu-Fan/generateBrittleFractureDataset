#include "particles.h"

// initialize MPM particles from mesh
void initParticlesFromMesh(std::vector<mpmParticle>& particles, Mesh& objMesh, int materialIndex, double density)
{
	objMesh.corresMPMPar_node.clear();
	int numMPMPar = particles.size();
	for (int i = 0; i < objMesh.pos_node.size(); i++)
	{
		mpmParticle par1;
		par1.position = objMesh.pos_node[i];
		par1.velocity = objMesh.vel_node[i];
		par1.volume = objMesh.vol_node[i];
		par1.material = materialIndex;
		par1.mass = density * par1.volume;
		par1.color = 0; 
		par1.ifCloth = true;
		par1.corresNode_Mesh = i;
		objMesh.corresMPMPar_node.push_back(numMPMPar);
		particles.push_back(par1);

		numMPMPar += 1;
	}
}


void calDeformedTriQuantities(std::vector<mpmParticle>& particles, Mesh& objMesh)
{
	// 1. calculate the triangle's normal
	objMesh.normal_deformed_triangle.clear();
	for (int i = 0; i < objMesh.triangles.size(); i++)
	{
		Eigen::Vector3i tri = objMesh.triangles[i];
		int vi_ = tri[0], vj_ = tri[1], vk_ = tri[2];
		int vi = objMesh.corresMPMPar_node[vi_], vj = objMesh.corresMPMPar_node[vj_], vk = objMesh.corresMPMPar_node[vk_];
		Eigen::Vector3d xi = particles[vi].position, xj = particles[vj].position, xk = particles[vk].position;

		// cclculate the triangle normal
		Eigen::Vector3d nijk = (xj - xi).cross(xk - xi).normalized();
		objMesh.normal_deformed_triangle.push_back(nijk);
	}

	// 2. calculate the triangle's mid-edge normal
	objMesh.midEdge_normal_deformed_triangle.clear();
	for (int i = 0; i < objMesh.triangles.size(); i++)
	{
		Eigen::Vector3i tri = objMesh.triangles[i];
		int vi_ = tri[0], vj_ = tri[1], vk_ = tri[2];
		int vi = objMesh.corresMPMPar_node[vi_], vj = objMesh.corresMPMPar_node[vj_], vk = objMesh.corresMPMPar_node[vk_];
		Eigen::Vector3d xi = particles[vi].position, xj = particles[vj].position, xk = particles[vk].position;

		// calculate the mid-edge normal
		std::string e0 = std::to_string(std::min(vi_, vj_)) + "#" + std::to_string(std::max(vi_, vj_));
		std::string e1 = std::to_string(std::min(vj_, vk_)) + "#" + std::to_string(std::max(vj_, vk_));
		std::string e2 = std::to_string(std::min(vk_, vi_)) + "#" + std::to_string(std::max(vk_, vi_));
		Eigen::Matrix3d midEdgeNorm = Eigen::Matrix3d::Zero();
		Eigen::Vector2i tris = objMesh.edgeTris[e0];
		if (tris[1] == -99)
		{
			midEdgeNorm.col(2) = objMesh.normal_deformed_triangle[i];
		}
		else
		{
			midEdgeNorm.col(2) = 0.5 * (objMesh.normal_deformed_triangle[tris[0]] + objMesh.normal_deformed_triangle[tris[1]]);
		}

		tris = objMesh.edgeTris[e1];
		if (tris[1] == -99)
		{
			midEdgeNorm.col(0) = objMesh.normal_deformed_triangle[i];
		}
		else
		{
			midEdgeNorm.col(0) = 0.5 * (objMesh.normal_deformed_triangle[tris[0]] + objMesh.normal_deformed_triangle[tris[1]]);
		}

		tris = objMesh.edgeTris[e2];
		if (tris[1] == -99)
		{
			midEdgeNorm.col(1) = objMesh.normal_deformed_triangle[i];
		}
		else
		{
			midEdgeNorm.col(1) = 0.5 * (objMesh.normal_deformed_triangle[tris[0]] + objMesh.normal_deformed_triangle[tris[1]]);
		}

		objMesh.midEdge_normal_deformed_triangle.push_back(midEdgeNorm);
	}

	// 3. the deformed triangle's quantities
	objMesh.l_triangle.clear();
	objMesh.t_triangle.clear();
	objMesh.q_triangle.clear();
	for (int i = 0; i < objMesh.triangles.size(); i++)
	{
		Eigen::Vector3i tri = objMesh.triangles[i];
		int vi_ = tri[0], vj_ = tri[1], vk_ = tri[2];
		int vi = objMesh.corresMPMPar_node[vi_], vj = objMesh.corresMPMPar_node[vj_], vk = objMesh.corresMPMPar_node[vk_];

		Eigen::Vector3d xi = particles[vi].position, xj = particles[vj].position, xk = particles[vk].position;
		Eigen::Vector3d ni = objMesh.midEdge_normal_deformed_triangle[i].col(0), nj = objMesh.midEdge_normal_deformed_triangle[i].col(1), nk = objMesh.midEdge_normal_deformed_triangle[i].col(2);
		Eigen::Vector3d nijk = objMesh.normal_deformed_triangle[i];


		Eigen::Matrix3d t = Eigen::Matrix3d::Zero(), q = Eigen::Matrix3d::Zero();
		Eigen::Matrix2d a = Eigen::Matrix2d::Zero(), b = Eigen::Matrix2d::Zero();


		t.col(0) = xj - xi;
		t.col(1) = xk - xi;
		t.col(2) = nijk;

		q.col(0) = 2.0 * (ni - nj);
		q.col(1) = 2.0 * (ni - nk);

		a(0, 0) = (xj - xi).dot(xj - xi);
		a(0, 1) = (xj - xi).dot(xk - xi);
		a(1, 0) = (xj - xi).dot(xk - xi);
		a(1, 1) = (xk - xi).dot(xk - xi);

		b(0, 0) = (ni - nj).dot(xi - xj);
		b(0, 1) = (ni - nj).dot(xi - xk);
		b(1, 0) = (ni - nk).dot(xi - xj);
		b(1, 1) = (ni - nk).dot(xi - xk);

		// undeformed curvature and determinant
		Eigen::Matrix2d l = a.inverse() * b;

		objMesh.l_triangle.push_back(l);
		objMesh.t_triangle.push_back(t);
		objMesh.q_triangle.push_back(q);

	}

	// 4. calculate the F0, F1, and F2
	objMesh.F0_triangle.clear();
	objMesh.F1_triangle.clear();
	objMesh.F2_triangle.clear();
	for (int i = 0; i < objMesh.triangles.size(); i++)
	{
		Eigen::Matrix3d t = objMesh.t_triangle[i];
		Eigen::Matrix3d T = objMesh.T_triangle[i];
		Eigen::Matrix3d q = objMesh.q_triangle[i];
		Eigen::Matrix3d Q = objMesh.Q_triangle[i];
		Eigen::Matrix2d l = objMesh.l_triangle[i];
		Eigen::Matrix2d l_bar = objMesh.l_bar_triangle[i];

		Eigen::Matrix3d	F0 = t * T.inverse();
		Eigen::Matrix3d	F1 = q * T.inverse() - t * T.inverse() * Q * T.inverse();
		Eigen::Matrix3d	F2 = t * (T.inverse() * Q) * (T.inverse() * Q) * T.inverse() - q * T.inverse() * Q * T.inverse();

		objMesh.F0_triangle.push_back(F0);
		objMesh.F1_triangle.push_back(F1);
		objMesh.F2_triangle.push_back(F2);


		{
			//Eigen::Matrix3d tmp1 = Eigen::Matrix3d::Zero();
			//tmp1.block<2, 2>(0, 0) = l_bar - l;
			//Eigen::Matrix3d	F1 = t * tmp1 * T.inverse();

			//Eigen::Matrix3d tmp2 = Eigen::Matrix3d::Zero();
			//tmp2.block<2, 2>(0, 0) = (l_bar - l) * l_bar;
			//Eigen::Matrix3d	F2 = t * tmp2 * T.inverse();
		}


	}

	// 5. calculate the partial derivative of ni, nj, nk and nijk wrt xi, xj, xk
	std::vector<Eigen::Matrix3d> vec_nijk_wrt_xi, vec_nijk_wrt_xj, vec_nijk_wrt_xk;
	for (int i = 0; i < objMesh.triangles.size(); i++)
	{
		// derivative of nijk
		Eigen::Vector3i tri = objMesh.triangles[i];
		int vi_ = tri[0], vj_ = tri[1], vk_ = tri[2];
		int vi = objMesh.corresMPMPar_node[vi_], vj = objMesh.corresMPMPar_node[vj_], vk = objMesh.corresMPMPar_node[vk_];
		Eigen::Vector3d xi = particles[vi].position, xj = particles[vj].position, xk = particles[vk].position;	
		Eigen::Vector3d nijk = objMesh.normal_deformed_triangle[i];
		
		Eigen::Vector3d a = xj - xi, b = xk - xi;
		Eigen::Matrix3d T = (Eigen::Matrix3d::Identity() - nijk * nijk.transpose()) / (a.cross(b).norm());
		Eigen::Matrix3d nijk_wrt_xi = T * (getSkewMatrix(b) - getSkewMatrix(a));
		Eigen::Matrix3d nijk_wrt_xj = -T * getSkewMatrix(b);
		Eigen::Matrix3d nijk_wrt_xk = T * getSkewMatrix(a);
		
		vec_nijk_wrt_xi.push_back(nijk_wrt_xi);
		vec_nijk_wrt_xj.push_back(nijk_wrt_xj);
		vec_nijk_wrt_xk.push_back(nijk_wrt_xk);
	}
	std::vector<Eigen::Matrix3d> vec_ni_wrt_xi, vec_ni_wrt_xj, vec_ni_wrt_xk;
	std::vector<Eigen::Matrix3d> vec_nj_wrt_xi, vec_nj_wrt_xj, vec_nj_wrt_xk;
	std::vector<Eigen::Matrix3d> vec_nk_wrt_xi, vec_nk_wrt_xj, vec_nk_wrt_xk;
	for (int i = 0; i < objMesh.triangles.size(); i++)
	{		
		Eigen::Vector3i tri = objMesh.triangles[i];
		int vi_ = tri[0], vj_ = tri[1], vk_ = tri[2];
		int vi = objMesh.corresMPMPar_node[vi_], vj = objMesh.corresMPMPar_node[vj_], vk = objMesh.corresMPMPar_node[vk_];
		Eigen::Vector3d xi = particles[vi].position, xj = particles[vj].position, xk = particles[vk].position;

		// derivative of ni
		{
			// calculate the mid-edge normal
			std::string ejk = std::to_string(std::min(vj_, vk_)) + "#" + std::to_string(std::max(vj_, vk_));
			Eigen::Vector2i tris = objMesh.edgeTris[ejk];

			
			if (tris[1] == -99) // edge jk is a boundary edge
			{		
				vec_ni_wrt_xi.push_back(vec_nijk_wrt_xi[i]);
				vec_ni_wrt_xj.push_back(vec_nijk_wrt_xj[i]); 
				vec_ni_wrt_xk.push_back(vec_nijk_wrt_xk[i]);
			}
			else
			{
				vec_ni_wrt_xi.push_back(0.5 * vec_nijk_wrt_xi[i]);

				int theOtherTriIndex = -99;
				if (tris[0] == i)
				{
					theOtherTriIndex = tris[1];
				}
				else
				{
					theOtherTriIndex = tris[0];
				}

				Eigen::Vector3i tri_other = objMesh.triangles[theOtherTriIndex];
				if (vj_ == tri_other[0])
				{
					vec_ni_wrt_xj.push_back(0.5 * vec_nijk_wrt_xj[i] + 0.5 * vec_nijk_wrt_xi[theOtherTriIndex]);
					if (vk_ == tri_other[1])
					{
						vec_ni_wrt_xk.push_back(0.5 * vec_nijk_wrt_xk[i] + 0.5 * vec_nijk_wrt_xj[theOtherTriIndex]);
					}
					else if(vk_ == tri_other[2])
					{
						vec_ni_wrt_xk.push_back(0.5 * vec_nijk_wrt_xk[i] + 0.5 * vec_nijk_wrt_xk[theOtherTriIndex]);
					}
				}
				else if (vj_ == tri_other[1])
				{
					vec_ni_wrt_xj.push_back(0.5 * vec_nijk_wrt_xj[i] + 0.5 * vec_nijk_wrt_xj[theOtherTriIndex]);
					if (vk_ == tri_other[0])
					{
						vec_ni_wrt_xk.push_back(0.5 * vec_nijk_wrt_xk[i] + 0.5 * vec_nijk_wrt_xi[theOtherTriIndex]);
					}
					else if (vk_ == tri_other[2])
					{
						vec_ni_wrt_xk.push_back(0.5 * vec_nijk_wrt_xk[i] + 0.5 * vec_nijk_wrt_xk[theOtherTriIndex]);
					}
				}
				else
				{
					vec_ni_wrt_xj.push_back(0.5 * vec_nijk_wrt_xj[i] + 0.5 * vec_nijk_wrt_xk[theOtherTriIndex]);
					if (vk_ == tri_other[0])
					{
						vec_ni_wrt_xk.push_back(0.5 * vec_nijk_wrt_xk[i] + 0.5 * vec_nijk_wrt_xi[theOtherTriIndex]);
					}
					else if (vk_ == tri_other[1])
					{
						vec_ni_wrt_xk.push_back(0.5 * vec_nijk_wrt_xk[i] + 0.5 * vec_nijk_wrt_xj[theOtherTriIndex]);
					}
				}


			}

		}


		// derivative of nj
		{
			// calculate the mid-edge normal
			std::string eik = std::to_string(std::min(vi_, vk_)) + "#" + std::to_string(std::max(vi_, vk_));
			Eigen::Vector2i tris = objMesh.edgeTris[eik];


			if (tris[1] == -99) // edge ik is a boundary edge
			{
				vec_nj_wrt_xi.push_back(vec_nijk_wrt_xi[i]);
				vec_nj_wrt_xj.push_back(vec_nijk_wrt_xj[i]);
				vec_nj_wrt_xk.push_back(vec_nijk_wrt_xk[i]);
			}
			else
			{
				vec_nj_wrt_xj.push_back(0.5 * vec_nijk_wrt_xj[i]);

				int theOtherTriIndex = -99;
				if (tris[0] == i)
				{
					theOtherTriIndex = tris[1];
				}
				else
				{
					theOtherTriIndex = tris[0];
				}

				Eigen::Vector3i tri_other = objMesh.triangles[theOtherTriIndex];
				if (vi_ == tri_other[0])
				{
					vec_nj_wrt_xi.push_back(0.5 * vec_nijk_wrt_xi[i] + 0.5 * vec_nijk_wrt_xi[theOtherTriIndex]);
					if (vk_ == tri_other[1])
					{
						vec_nj_wrt_xk.push_back(0.5 * vec_nijk_wrt_xk[i] + 0.5 * vec_nijk_wrt_xj[theOtherTriIndex]);
					}
					else if (vk_ == tri_other[2])
					{
						vec_nj_wrt_xk.push_back(0.5 * vec_nijk_wrt_xk[i] + 0.5 * vec_nijk_wrt_xk[theOtherTriIndex]);
					}
				}
				else if (vi_ == tri_other[1])
				{
					vec_nj_wrt_xi.push_back(0.5 * vec_nijk_wrt_xi[i] + 0.5 * vec_nijk_wrt_xj[theOtherTriIndex]);
					if (vk_ == tri_other[0])
					{
						vec_nj_wrt_xk.push_back(0.5 * vec_nijk_wrt_xk[i] + 0.5 * vec_nijk_wrt_xi[theOtherTriIndex]);
					}
					else if (vk_ == tri_other[2])
					{
						vec_nj_wrt_xk.push_back(0.5 * vec_nijk_wrt_xk[i] + 0.5 * vec_nijk_wrt_xk[theOtherTriIndex]);
					}
				}
				else
				{
					vec_nj_wrt_xi.push_back(0.5 * vec_nijk_wrt_xi[i] + 0.5 * vec_nijk_wrt_xk[theOtherTriIndex]);
					if (vk_ == tri_other[0])
					{
						vec_nj_wrt_xk.push_back(0.5 * vec_nijk_wrt_xk[i] + 0.5 * vec_nijk_wrt_xi[theOtherTriIndex]);
					}
					else if (vk_ == tri_other[1])
					{
						vec_nj_wrt_xk.push_back(0.5 * vec_nijk_wrt_xk[i] + 0.5 * vec_nijk_wrt_xj[theOtherTriIndex]);
					}
				}


			}

		}



		// derivative of nk
		{
			// calculate the mid-edge normal
			std::string eij = std::to_string(std::min(vi_, vj_)) + "#" + std::to_string(std::max(vi_, vj_));
			Eigen::Vector2i tris = objMesh.edgeTris[eij];


			if (tris[1] == -99) // edge ik is a boundary edge
			{
				vec_nk_wrt_xi.push_back(vec_nijk_wrt_xi[i]);
				vec_nk_wrt_xj.push_back(vec_nijk_wrt_xj[i]);
				vec_nk_wrt_xk.push_back(vec_nijk_wrt_xk[i]);
			}
			else
			{
				vec_nk_wrt_xk.push_back(0.5 * vec_nijk_wrt_xj[i]);

				int theOtherTriIndex = -99;
				if (tris[0] == i)
				{
					theOtherTriIndex = tris[1];
				}
				else
				{
					theOtherTriIndex = tris[0];
				}

				Eigen::Vector3i tri_other = objMesh.triangles[theOtherTriIndex];
				if (vi_ == tri_other[0])
				{
					vec_nk_wrt_xi.push_back(0.5 * vec_nijk_wrt_xi[i] + 0.5 * vec_nijk_wrt_xi[theOtherTriIndex]);
					if (vj_ == tri_other[1])
					{
						vec_nk_wrt_xj.push_back(0.5 * vec_nijk_wrt_xj[i] + 0.5 * vec_nijk_wrt_xj[theOtherTriIndex]);
					}
					else if (vj_ == tri_other[2])
					{
						vec_nk_wrt_xj.push_back(0.5 * vec_nijk_wrt_xj[i] + 0.5 * vec_nijk_wrt_xk[theOtherTriIndex]);
					}
				}
				else if (vi_ == tri_other[1])
				{
					vec_nk_wrt_xi.push_back(0.5 * vec_nijk_wrt_xi[i] + 0.5 * vec_nijk_wrt_xj[theOtherTriIndex]);
					if (vj_ == tri_other[0])
					{
						vec_nk_wrt_xj.push_back(0.5 * vec_nijk_wrt_xj[i] + 0.5 * vec_nijk_wrt_xi[theOtherTriIndex]);
					}
					else if (vj_ == tri_other[2])
					{
						vec_nk_wrt_xj.push_back(0.5 * vec_nijk_wrt_xj[i] + 0.5 * vec_nijk_wrt_xk[theOtherTriIndex]);
					}
				}
				else
				{
					vec_nk_wrt_xi.push_back(0.5 * vec_nijk_wrt_xi[i] + 0.5 * vec_nijk_wrt_xk[theOtherTriIndex]);
					if (vj_ == tri_other[0])
					{
						vec_nk_wrt_xj.push_back(0.5 * vec_nijk_wrt_xj[i] + 0.5 * vec_nijk_wrt_xi[theOtherTriIndex]);
					}
					else if (vj_ == tri_other[1])
					{
						vec_nk_wrt_xj.push_back(0.5 * vec_nijk_wrt_xj[i] + 0.5 * vec_nijk_wrt_xj[theOtherTriIndex]);
					}
				}


			}

		}


	}

	// 6. calculate the partial derivative of t and q wrt xi, xj, xk



}