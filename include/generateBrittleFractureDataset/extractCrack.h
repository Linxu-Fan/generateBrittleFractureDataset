#ifndef EXTRACTCRACK_H

#define EXTRACTCRACK_H

#include "utils.h"
#include "weights.h"
#include "damageGradient.h"
#include "voro++.hh"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <set>
#include <sstream>
#include <string>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "omp.h"
#include <vector>
#include <math.h>
#include <map>
#include <assert.h> 
#include <random>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <cfloat>


namespace extractCrackSurface
{
    // Struct of particles
    struct Particle 
    {
        Eigen::Vector3d pos = { 0, 0, 0 }, vel = { 0, 0, 0 }; // each particle's x and y position , velocity, and momentum
        double m = 0; // each particle's mass
        Eigen::Vector3i ppIndex = { 0, 0, 0 }; // particle base index
        Eigen::MatrixXd weight;
        Eigen::MatrixXd deltaWeight;
        double Dp = 0; // particle's scalar damage value
        Eigen::Vector3d deltaD = { 0, 0, 0 }; // particle's damage gradient
        int color = 0; // boundary that traction is applied

        Particle(Eigen::Vector3d ipos, Eigen::Vector3d ivel, double im, int ic, double iDp)
            : pos(ipos)
            , vel(ivel)
            , m(im)
            , color(ic)
            , Dp(iDp)
        {
        }
    };


    struct parametersSim {

        // computational domain
        Eigen::Vector3d length = { 1, 1, 1 }; // computation cube lengths of three dimensions (x , y , z). The origin point is (0 , 0 , 0)
        Eigen::Vector3d minCoordinate = { 0, 0, 0 }; // the minimum coordinate of the computation domain

        // Bcakground Eulerian grid
        double dx = 2E-2;

        // openvdb voxel size
        double vdbVoxelSize = 0.0005;

        // applied force
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> appliedForce;

        double damageThreshold = 0.97; // after this threshold,
    };

    static int calculateID(int x, int y, int z, Eigen::Vector3d len, double dx) // coordinate of x and y, length in three dimensions of the cube, grid space
    {
        Eigen::Vector3i length = (len / dx).cast<int>() + Eigen::Vector3i::Constant(1);
        int ID = z * (length(0) * length(1)) + y * length(0) + x;
        return ID;
    };

    struct meshObjFormat {
        // input mesh vertices and faces
        std::vector<Eigen::Vector3d> vertices;
        std::vector<std::vector<int>> faces;
        std::vector<int> faceFromVoroCell; // indicate the voronoi cell which the face belongs to
        std::vector<int> faceFromtheOtherVoroCell; // indicate the voronoi cell which the face belongs to(the other side)
    };


    // Struct of grid
    struct Grid {
        // property of velocity-field 0
        double m = 0; // each node's mass
        Eigen::Vector3d mom = { 0, 0, 0 }; // each node's momentum
        Eigen::Vector3d velocity = { 0, 0, 0 }; // each node's velocity
        Eigen::Vector3d force = { 0, 0, 0 }; // each node's force

        // general grid node property
        Eigen::Vector3i posIndex = { 0, 0, 0 };
        Eigen::Vector3d deltaDi = { 0, 0, 0 }; // gradient of damage field
        double Di = 0; // value of damage field
        double sw = 0; // sum of particle-grid weight

        // particle index in the support radius of this node. The order of the vector is important
        std::vector<int> supportParticles; // store the position of the particle in vector "particles";
        std::vector<double> supportParticlesWeight; // store the weight of particle to the grid node

        // set of crack surface points withing the grid cell
        std::vector<int> crackPoints;
        int nearestPoint = -1000; // (nearestPoint < 0) means it is far away from the crack surface
        Eigen::Vector3d crackSurfaceNormal = { 0, 0, 0 }; // the vector pointing from the nearest point on the crack surface to the grid node

        // parameters of contact algorithm
        double mass_0 = 0;
        Eigen::Vector3d mom_0 = { 0, 0, 0 };
        Eigen::Vector3d velocity_0 = { 0, 0, 0 };
        Eigen::Vector3d force_0 = { 0, 0, 0 };

        double mass_1 = 0;
        Eigen::Vector3d mom_1 = { 0, 0, 0 };
        Eigen::Vector3d velocity_1 = { 0, 0, 0 };
        Eigen::Vector3d force_1 = { 0, 0, 0 };

        Grid(double im)
            : m(im)
        {
        }
    };


    // Struct of particles
    struct Point {
        int index = 0; // index of each point
        Eigen::Vector3d pos = { 0, 0, 0 }; // each point's position
        int numVertices = 0; // number of vertices
        std::vector<Eigen::Vector3d> verticsCoor; // vertices' coordinate
        int numFaces = 0; // number of faces
        std::vector<std::vector<int>> verticesFace; // vertices of each face
        std::vector<Eigen::Vector3d> surfaceNormal; // vertices' coordinate
        std::vector<int> neighbour; // neighbour points that share common faces with this point
        std::vector<int> neighbourCalculated; // neighbour points that have already find shared face

        std::vector<int> neighbourSameSide; // neighbour points that are on the same side with this point
        std::vector<int> neighbourOtherSide; // neighbour points that are on the other side with this point


        std::vector<int> globalIndexed; // if a vertex finds a global index or not. If yes, the index is a value lager than 0; If not, it is -99;
        std::vector<bool> faceIndexed; // if a face is indexed or not;


        Point(int iindex, Eigen::Vector3d ipos, int inumVertices, std::vector<Eigen::Vector3d> iverticsCoor, int inumFaces, std::vector<std::vector<int>> iverticesFace, std::vector<Eigen::Vector3d> isurfaceNormal, std::vector<int> ineighbour)
            : index(iindex)
            , pos(ipos)
            , numVertices(inumVertices)
            , verticsCoor(iverticsCoor)
            , numFaces(inumFaces)
            , verticesFace(iverticesFace)
            , surfaceNormal(isurfaceNormal)
            , neighbour(ineighbour)
        {
        }
    };



    // calculate the damage gradient of all particles and grid nodes.
    static void calDamageGradient(std::vector<Particle>* particles, parametersSim param, double dx, std::map<int, int>* gridMap, std::vector<Grid>* nodesVec)
    {
        for (int f = 0; f < particles->size(); f++) {
            struct weightAndDreri WD = calWeight(dx, (*particles)[f].pos);

            (*particles)[f].ppIndex = WD.ppIndex;
            (*particles)[f].weight = WD.weight;
            (*particles)[f].deltaWeight = WD.deltaWeight;
        };

        int count = -1; // count the number of active grid node
        // number of grid nodes per edge

        // calculate node damage value
        for (int f = 0; f < particles->size(); f++) {
            (*particles)[f].deltaD = { 0, 0, 0 };

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        int ID = calculateID((*particles)[f].ppIndex[0] + i, (*particles)[f].ppIndex[1] + j, (*particles)[f].ppIndex[2] + k, param.length, dx);
                        double weight = (*particles)[f].weight(0, i) * (*particles)[f].weight(1, j) * (*particles)[f].weight(2, k);

                        if (weight != 0) {
                            if ((*gridMap).find(ID) == (*gridMap).end()) {
                                count += 1;
                                (*gridMap)[ID] = count;
                                (*nodesVec).push_back(Grid(0.0));

                                (*nodesVec)[count].posIndex = { (*particles)[f].ppIndex[0] + i, (*particles)[f].ppIndex[1] + j, (*particles)[f].ppIndex[2] + k };
                                (*nodesVec)[count].Di += (*particles)[f].Dp * weight;
                                (*nodesVec)[count].sw += weight;
                            }
                            else {
                                int eid = (*gridMap)[ID];

                                (*nodesVec)[eid].Di += (*particles)[f].Dp * weight;
                                (*nodesVec)[eid].sw += weight;
                            };
                        };
                    };
                };
            };
        };

        // calculate particle damage gradient
        for (int f = 0; f < particles->size(); f++) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        int ID = calculateID((*particles)[f].ppIndex[0] + i, (*particles)[f].ppIndex[1] + j, (*particles)[f].ppIndex[2] + k, param.length, dx);
                        double weight = (*particles)[f].weight(0, i) * (*particles)[f].weight(1, j) * (*particles)[f].weight(2, k);

                        if (weight != 0) {
                            int eid = (*gridMap)[ID];

                            Eigen::Vector3d posD = (*particles)[f].pos - (*nodesVec)[eid].posIndex.cast<double>() * dx;
                            (*particles)[f].deltaD += weight / (dx * dx / 4) * (*nodesVec)[eid].Di / (*nodesVec)[eid].sw * posD;
                        };
                    };
                };
            };
        };

        // calculate grid node's damage gradient. This gives the exact value.
        for (int g = 0; g < (*nodesVec).size(); g++) {
            Eigen::Vector3d weightVec = { 0.125, 0.75, 0.125 };
            Eigen::Vector3d posVec = { dx, 0, -dx };

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        int ID = calculateID((int)(*nodesVec)[g].posIndex[0] + i - 1, (int)(*nodesVec)[g].posIndex[1] + j - 1, (int)(*nodesVec)[g].posIndex[2] + k - 1, param.length, dx);
                        double weight = weightVec[i] * weightVec[j] * weightVec[k];

                        if ((*gridMap).find(ID) != (*gridMap).end()) {
                            int eid = (*gridMap)[ID];
                            Eigen::Vector3d posD = { posVec[i], posVec[j], posVec[k] };
                            (*nodesVec)[g].deltaDi += weight / (dx * dx / 4) * (*nodesVec)[eid].Di / (*nodesVec)[eid].sw * posD;
                        };
                    };
                };
            };

            (*nodesVec)[g].Di = (*nodesVec)[g].Di / (*nodesVec)[g].sw;
        };
    };

    // calculate the damage gradient of any give point
    static Eigen::Vector3d calDamageGradientPoint(Eigen::Vector3d pos, parametersSim param, double dx, std::map<int, int>* gridMap, std::vector<Grid>* nodesVec)
    {
        Eigen::Vector3d deltaPoint = { 0, 0, 0 };
        Eigen::Vector3d base = (pos) / dx - Eigen::Vector3d::Constant(0.5);
        Eigen::Vector3i ppIndex = base.cast<int>();

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    int ID = calculateID(ppIndex[0] + i, ppIndex[1] + j, ppIndex[2] + k, param.length, dx);

                    struct weightAndDreri WD = calWeight(dx, (pos));
                    Eigen::MatrixXd weightPoint = WD.weight;
                    double weight = weightPoint(0, i) * weightPoint(1, j) * weightPoint(2, k);

                    if ((*gridMap).find(ID) != (*gridMap).end()) {
                        int eid = (*gridMap)[ID];
                        Eigen::Vector3d posD = (pos)-(*nodesVec)[eid].posIndex.cast<double>() * dx;
                        deltaPoint += weight / (dx * dx / 4) * (*nodesVec)[eid].Di / (*nodesVec)[eid].sw * posD;
                    };
                };
            };
        };

        return deltaPoint;
    };

    // calculate the damage value of any give point
    static double calDamageValuePoint(Eigen::Vector3d pos, parametersSim param, double dx, std::map<int, int>* gridMap, std::vector<Grid>* nodesVec)
    {
        double dpValue = 0;
        Eigen::Vector3d base = (pos) / dx - Eigen::Vector3d::Constant(0.5);
        Eigen::Vector3i ppIndex = base.cast<int>();

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    int ID = calculateID(ppIndex[0] + i, ppIndex[1] + j, ppIndex[2] + k, param.length, dx);

                    struct weightAndDreri WD = calWeight(dx, (pos));
                    Eigen::MatrixXd weightPoint = WD.weight;
                    double weight = weightPoint(0, i) * weightPoint(1, j) * weightPoint(2, k);

                    if ((*gridMap).find(ID) != (*gridMap).end()) {
                        int eid = (*gridMap)[ID];
                        Eigen::Vector3d posD = (pos)-(*nodesVec)[eid].posIndex.cast<double>() * dx;
                        dpValue += weight * (*nodesVec)[eid].Di;
                    };
                };
            };
        };

        return dpValue;
    };



    // read obj file
    static struct meshObjFormat readObj(std::string path)
    {
        meshObjFormat result;

        // input mesh vertices and faces
        std::vector<Eigen::Vector3d> vertices;
        std::vector<std::vector<int>> faces;

        std::ifstream in;
        in.open(path);
        std::string line;
        while (getline(in, line)) {
            if (line.size() > 0) {
                std::vector<std::string> vecCoor = split(line, " ");
                if (vecCoor.size() == 0) {
                    std::cout << "Obj mesh read error!" << std::endl;
                }
                if (vecCoor[0] == "v") {
                    Eigen::Vector3d vertex = { std::stod(vecCoor[1]), std::stod(vecCoor[2]), std::stod(vecCoor[3]) };
                    vertices.push_back(vertex);
                }
                if (vecCoor[0] == "f") {
                    std::vector<int> face;

                    for (int k = 1; k < vecCoor.size(); k++) {
                        //std::cout<<k<<": " <<vecCoor[k]<<std::endl;
                        face.push_back(std::stoi(vecCoor[k]) - 1);
                    }
                    faces.push_back(face);
                }
            }
        }
        in.close();

        result.faces = faces;
        result.vertices = vertices;

        return result;
    }

    // split a line from a text file
    static std::vector<std::string> split(const std::string& s, const std::string& seperator)
    {
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

    // Set the damage phase of a grid node vector into a specific value
    static void setNodeValue(std::vector<Grid>* nodesVec, int va)
    {
        int numDamage = (*nodesVec).size();
        for (int k = 0; k < numDamage; k++) {
            (*nodesVec)[k].Di = va;
        }
    }

    // Find the bounding box boundary nodes and set its damage phase into a specific value
    static void findBoundaryNodes(std::vector<Particle>* particles, std::vector<Grid>* nodesVec, std::map<int, int>* gridMap, struct parametersSim parti, int va)
    {

        int count1 = (*nodesVec).size();
        int numDamage = (*nodesVec).size();
        for (int m = 0; m < numDamage; m++) {
            for (int i = -1; i < 2; i++) {
                for (int j = -1; j < 2; j++) {
                    for (int k = -1; k < 2; k++) {

                        int ID = calculateID((*nodesVec)[m].posIndex[0] + i, (*nodesVec)[m].posIndex[1] + j, (*nodesVec)[m].posIndex[2] + k, parti.length, parti.dx);
                        if ((*gridMap).find(ID) == (*gridMap).end()) {
                            (*gridMap)[ID] = count1;
                            (*nodesVec).push_back(Grid(0.0));

                            (*nodesVec)[count1].posIndex = { (*nodesVec)[m].posIndex[0] + i, (*nodesVec)[m].posIndex[1] + j, (*nodesVec)[m].posIndex[2] + k };
                            (*nodesVec)[count1].Di = va;

                            count1 += 1;
                        }
                    }
                }
            }
        }
    }

    // Read particles' positions and damage phases
    static void readParticles(std::vector<Particle>* particlesRaw, std::vector<Particle>* particles, bool ifFully, struct parametersSim param)
    {

        for (int i = 0; i < (*particlesRaw).size(); i++) {
            if (ifFully == true) {
                if ((*particlesRaw)[i].Dp >= param.damageThreshold) {
                    (*particles).push_back(Particle((*particlesRaw)[i].pos, (*particlesRaw)[i].vel, 0, 0, 1));
                }
            }
            else {
                (*particles).push_back(Particle((*particlesRaw)[i].pos, (*particlesRaw)[i].vel, 0, 0, 1));
            }
        }
    }

    // Calculate the damage value of any point and return the value
    static double ifFullyDamaged(Eigen::Vector3d pos, parametersSim param, std::map<int, int>* gridMap, std::vector<Grid>* nodesVec)
    {
        double damageValue = 0;
        Eigen::Vector3d base = pos / param.dx - Eigen::Vector3d::Constant(0.5);
        Eigen::Vector3i ppIndex = base.cast<int>();

        int countFullyDamaged = 0;

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    int ID = calculateID(ppIndex[0] + i, ppIndex[1] + j, ppIndex[2] + k, param.length, param.dx);
                    struct weightAndDreri WD = calWeight(param.dx, pos);
                    Eigen::MatrixXd weightPoint = WD.weight;
                    double weight = weightPoint(0, i) * weightPoint(1, j) * weightPoint(2, k);

                    if ((*gridMap).find(ID) != (*gridMap).end()) {
                        int eid = (*gridMap)[ID];
                        damageValue += weight * (*nodesVec)[eid].Di;
                        if ((*nodesVec)[eid].Di == 1) {
                            countFullyDamaged += 1;
                        }
                    }
                };
            };
        };

        if (countFullyDamaged >= 8) {
            damageValue = 1.0;
        }

        return damageValue;
    }

    // Store the index of each vertex and its position
    static int findIndexVertex(Eigen::Vector3d pos, std::vector<Eigen::Vector3d>* vertexIndex)
    {
        int index = vertexIndex->size();
        (*vertexIndex).push_back(pos);

        return index;
    }

    // Read all structured nodes and calculate the damage gradient
    static void readParticlesAndCalGradient(std::vector<Grid>* fullyDamagedParticlesNodesVec, std::vector<Particle>* particles, parametersSim param, std::map<int, int>* gridMap, std::vector<Grid>* nodesVec)
    {
        for (int i = 0; i < (*fullyDamagedParticlesNodesVec).size(); i++) {
            if ((*fullyDamagedParticlesNodesVec)[i].Di == 1) {
                Eigen::Vector3d ipos = { (*fullyDamagedParticlesNodesVec)[i].posIndex[0] * param.dx, (*fullyDamagedParticlesNodesVec)[i].posIndex[1] * param.dx, (*fullyDamagedParticlesNodesVec)[i].posIndex[2] * param.dx };
                Eigen::Vector3d ivel = { 0, 0, 0 };
                (*particles).push_back(Particle(ipos, ivel, 0, 0, 1));
            }

            if ((*fullyDamagedParticlesNodesVec)[i].Di == 2) {
                Eigen::Vector3d ipos = { (*fullyDamagedParticlesNodesVec)[i].posIndex[0] * param.dx, (*fullyDamagedParticlesNodesVec)[i].posIndex[1] * param.dx, (*fullyDamagedParticlesNodesVec)[i].posIndex[2] * param.dx };
                Eigen::Vector3d ivel = { 0, 0, 0 };
                (*particles).push_back(Particle(ipos, ivel, 0, 0, 0));
            }
        }

        calDamageGradient(particles, param, param.dx, gridMap, nodesVec);
    }

    // Find paths between two nodes
    static bool findPath(Eigen::Vector3d startNode, Eigen::Vector3d stopNode, parametersSim param, std::vector<Grid>* fullyDamagedParticlesNodesVec, std::map<int, int>* fullyDamagedParticlesGridMap, std::vector<int>* surfaceNodesID)
    {

        Eigen::Vector3d startNodePos = startNode / param.dx;
        Eigen::Vector3i startNodeIndex;
        startNodeIndex[0] = round(startNodePos[0]);
        startNodeIndex[1] = round(startNodePos[1]);
        startNodeIndex[2] = round(startNodePos[2]);
        int startNodeID = calculateID(startNodeIndex[0], startNodeIndex[1], startNodeIndex[2], param.length, param.dx);

        Eigen::Vector3d stopNodePos = stopNode / param.dx;
        Eigen::Vector3i stopNodeIndex;
        stopNodeIndex[0] = round(stopNodePos[0]);
        stopNodeIndex[1] = round(stopNodePos[1]);
        stopNodeIndex[2] = round(stopNodePos[2]);
        int stopNodeID = calculateID(stopNodeIndex[0], stopNodeIndex[1], stopNodeIndex[2], param.length, param.dx);

        Eigen::Vector3i boundingBox = stopNodeIndex - startNodeIndex;
        int maxAllowedSteps = abs(boundingBox[0]) + abs(boundingBox[1]) + abs(boundingBox[2]);

        // if a particle has no neighbours
        bool noNeighbours = false;

        std::vector<Eigen::Vector3i> visitQueueIndex; // stores node posIndexs that should be visited
        std::map<int, int> IDMap; // the key is the ID of a node, the value is the node's position in std::vector nodeNeighbour
        std::vector<std::vector<int>> nodeParent; // stores parent-layer neighbouring information of nodes

        // Initialize all vectors with starting node
        visitQueueIndex.push_back(startNodeIndex);
        IDMap[startNodeID] = 0;
        std::vector<int> startNodeParent;
        startNodeParent.push_back(-99);
        nodeParent.push_back(startNodeParent); // start node has no parent-layer, so give a negative value

        bool reachMaxmiumBoundingBox = false; // if the expansion layer reach the maximum bounding box
        bool reachStopNode = false;
        int lastLayerPos = 0, lastLayerLength = 1;

        //////////////
        int depth = 0; // the number of step used

        do {

            std::vector<Eigen::Vector3i> visitQueueIndexLayer;
            std::vector<std::vector<int>> nodeParentLayer;
            int lengthOfALayer = 0;

            for (int k = lastLayerPos; k < lastLayerPos + lastLayerLength; k++) {
                Eigen::Vector3i parentNodeIndex = visitQueueIndex[k]; // the node that is going to be visited
                int parentNodeID = calculateID(parentNodeIndex[0], parentNodeIndex[1], parentNodeIndex[2], param.length, param.dx);

                for (int axis = 0; axis < 3; axis++) // three axis directions
                {
                    for (int pn = 0; pn < 2; pn++) {
                        Eigen::Vector3i normal = { 0, 0, 0 };
                        normal[axis] = 2 * pn - 1;
                        Eigen::Vector3i neighbourNodePosIndex = parentNodeIndex + normal;

                        int neighbourNodeID = calculateID(neighbourNodePosIndex[0], neighbourNodePosIndex[1], neighbourNodePosIndex[2], param.length, param.dx);
                        if (IDMap.find(neighbourNodeID) == IDMap.end()) // this node is not in the queue
                        {
                            if ((*fullyDamagedParticlesGridMap).find(neighbourNodeID) != (*fullyDamagedParticlesGridMap).end()) {
                                int eid = (*fullyDamagedParticlesGridMap)[neighbourNodeID];
                                if ((*fullyDamagedParticlesNodesVec)[eid].Di == 2) // if this node is a boundary shell
                                {
                                    visitQueueIndexLayer.push_back(neighbourNodePosIndex);
                                    IDMap[neighbourNodeID] = lengthOfALayer + lastLayerPos + lastLayerLength;

                                    // find the neghbours of this node in the last layer
                                    std::vector<int> aSingleNodeParent;
                                    for (int nt = lastLayerPos; nt < lastLayerPos + lastLayerLength; nt++) {
                                        Eigen::Vector3i lastLayerNode = visitQueueIndex[nt];
                                        Eigen::Vector3i diffIndex = lastLayerNode - neighbourNodePosIndex;
                                        int sumDiff = abs(diffIndex[0]) + abs(diffIndex[1]) + abs(diffIndex[2]);
                                        if (sumDiff == 1) {
                                            int lastLayerNodeID = calculateID(lastLayerNode[0], lastLayerNode[1], lastLayerNode[2], param.length, param.dx);
                                            aSingleNodeParent.push_back(lastLayerNodeID);
                                        }
                                    }
                                    nodeParentLayer.push_back(aSingleNodeParent);
                                    lengthOfALayer += 1;

                                    if (neighbourNodeID == stopNodeID) {
                                        reachStopNode = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            lastLayerPos = visitQueueIndex.size();
            lastLayerLength = lengthOfALayer;
            visitQueueIndex.insert(visitQueueIndex.end(), visitQueueIndexLayer.begin(), visitQueueIndexLayer.end());
            nodeParent.insert(nodeParent.end(), nodeParentLayer.begin(), nodeParentLayer.end());

            depth += 1;

            if (lastLayerLength == 0) {
                return true;
            }

            if (depth >= round(1.3 * (double)maxAllowedSteps) && reachStopNode == false) {
                return true;
            }

        } while (reachStopNode == false && noNeighbours == false);

        //// Finding all possible paths
        std::vector<std::vector<int>> paths;
        bool reachStartNode = false;

        std::vector<int> initializeStop;
        initializeStop.push_back(stopNodeID);
        paths.push_back(initializeStop);
        int pathSize = paths.size();

        do {
            int nextLayerSize = 0;
            for (int k = 0; k < pathSize; k++) {
                std::vector<int> partialPath = paths[k];
                std::vector<int> lastNodeParents = nodeParent[IDMap[paths[k].back()]];

                for (int m = 0; m < lastNodeParents.size(); m++) {

                    if (count((*surfaceNodesID).begin(), (*surfaceNodesID).end(), lastNodeParents[m]) == 0) {
                        //std::vector<int> partialPathExtend = partialPath;
                        std::vector<int> partialPathExtend;
                        partialPathExtend.push_back(lastNodeParents[m]);

                        if (lastNodeParents[m] == startNodeID) {
                            reachStartNode = true;
                            return false;
                        }

                        paths.push_back(partialPathExtend);
                        nextLayerSize += 1;
                    }
                }
            }

            paths.erase(paths.begin(), paths.begin() + pathSize);
            pathSize = nextLayerSize;
            //cout <<"pathSize = "<< pathSize << endl;

        } while (reachStartNode == false && pathSize != 0);

        if (reachStartNode == false) {
            return true;
        }
    }

    // Find if a pair of nodes belong to critical nodes. The function return true if one node is a critical node
    static bool ifCriticalNode(Eigen::Vector3d node1, parametersSim param, std::vector<int>* criticalNodeIndex)
    {

        Eigen::Vector3d node1Pos = node1 / param.dx;
        Eigen::Vector3i node1Index;
        node1Index[0] = round(node1Pos[0]);
        node1Index[1] = round(node1Pos[1]);
        node1Index[2] = round(node1Pos[2]);
        int node1ID = calculateID(node1Index[0], node1Index[1], node1Index[2], param.length, param.dx);

        if (count((*criticalNodeIndex).begin(), (*criticalNodeIndex).end(), node1ID) != 0) {
            return true;
        }
        else {
            return false;
        }
    }

    // Find the nearest boundary node of a critical node
    static Eigen::Vector3i findNearestBoundaryNode(int nodeIDPoint, std::vector<Point>* points, std::vector<int>* boundaryNodesID, std::vector<Eigen::Vector3i>* boundaryNodesPosIndex, std::map<int, int>* pointIndexFind, parametersSim param, std::vector<int>* criticalNodeIndex)
    {
        Eigen::Vector3i nodeIndex = (*boundaryNodesPosIndex)[nodeIDPoint];
        int nodeID = (*boundaryNodesID)[nodeIDPoint];

        bool reachNearest = false;

        std::vector<Eigen::Vector3i> visitQueueIndex; // stores node posIndexs that should be visited
        std::map<int, int> IDMap; // the key is the ID of a node, the value is the node's position in std::vector nodeNeighbour
        std::vector<std::vector<int>> nodeParent; // stores parent-layer neighbouring information of nodes

        // Initialize all vectors with starting node
        visitQueueIndex.push_back(nodeIndex);
        IDMap[nodeID] = 0;
        std::vector<int> startNodeParent;
        startNodeParent.push_back(-99);
        nodeParent.push_back(startNodeParent); // start node has no parent-layer, so give a negative value

        bool reachMaxmiumBoundingBox = false; // if the expansion layer reach the maximum bounding box
        bool reachStopNode = false;
        int lastLayerPos = 0, lastLayerLength = 1;

        do {

            std::vector<Eigen::Vector3i> visitQueueIndexLayer;
            std::vector<std::vector<int>> nodeParentLayer;
            int lengthOfALayer = 0;

            for (int k = lastLayerPos; k < lastLayerPos + lastLayerLength; k++) {
                Eigen::Vector3i parentNodeIndex = visitQueueIndex[k]; // the node that is going to be visited
                int parentNodeID = calculateID(parentNodeIndex[0], parentNodeIndex[1], parentNodeIndex[2], param.length, param.dx);

                for (int axis = 0; axis < 3; axis++) // three axis directions
                {
                    for (int pn = 0; pn < 2; pn++) {
                        Eigen::Vector3i normal = { 0, 0, 0 };
                        normal[axis] = 2 * pn - 1;
                        Eigen::Vector3i neighbourNodePosIndex = parentNodeIndex + normal;

                        int neighbourNodeID = calculateID(neighbourNodePosIndex[0], neighbourNodePosIndex[1], neighbourNodePosIndex[2], param.length, param.dx);
                        if (IDMap.find(neighbourNodeID) == IDMap.end()) // this node is not in the queue
                        {

                            if (count((*boundaryNodesID).begin(), (*boundaryNodesID).end(), neighbourNodeID) != 0) {
                                visitQueueIndexLayer.push_back(neighbourNodePosIndex);
                                IDMap[neighbourNodeID] = lengthOfALayer + lastLayerPos + lastLayerLength;

                                // find the neghbours of this node in the last layer
                                std::vector<int> aSingleNodeParent;
                                for (int nt = lastLayerPos; nt < lastLayerPos + lastLayerLength; nt++) {
                                    Eigen::Vector3i lastLayerNode = visitQueueIndex[nt];
                                    Eigen::Vector3i diffIndex = lastLayerNode - neighbourNodePosIndex;
                                    int sumDiff = abs(diffIndex[0]) + abs(diffIndex[1]) + abs(diffIndex[2]);
                                    if (sumDiff == 1) {
                                        int lastLayerNodeID = calculateID(lastLayerNode[0], lastLayerNode[1], lastLayerNode[2], param.length, param.dx);
                                        aSingleNodeParent.push_back(lastLayerNodeID);
                                    }
                                }
                                nodeParentLayer.push_back(aSingleNodeParent);
                                lengthOfALayer += 1;

                                if (ifCriticalNode(neighbourNodePosIndex.cast<double>() * param.dx, param, criticalNodeIndex) == false) {
                                    return neighbourNodePosIndex;
                                }
                            }
                        }
                    }
                }
            }
            lastLayerPos = visitQueueIndex.size();
            lastLayerLength = lengthOfALayer;
            visitQueueIndex.insert(visitQueueIndex.end(), visitQueueIndexLayer.begin(), visitQueueIndexLayer.end());
            nodeParent.insert(nodeParent.end(), nodeParentLayer.begin(), nodeParentLayer.end());

            // if this point is a desolate critical point, return a negative value
            if (lastLayerLength == 0) {
                Eigen::Vector3i noConnection = { -999, 0, 0 };
                return noConnection;
            }

        } while (reachNearest == false);
    }

    // Judge if a pair of points are on different sides of a crack
    static bool ifTwoSides(int startNode, int stopNode, std::vector<Point>* points, std::vector<int>* boundaryNodesID, std::vector<Eigen::Vector3i>* boundaryNodesPosIndex, std::map<int, int>* pointIndexFind, parametersSim param, std::vector<int>* criticalNodeIndex)
    {

        Eigen::Vector3i startNodeIndex = (*boundaryNodesPosIndex)[startNode];
        startNodeIndex = findNearestBoundaryNode(startNode, points, boundaryNodesID, boundaryNodesPosIndex, pointIndexFind, param, criticalNodeIndex);
        // if this point is a desolate critical point, return true. Keep this face though it may become a tooth.
        if (startNodeIndex[0] < 0) {
            return true;
        }
        int startNodeID = calculateID(startNodeIndex[0], startNodeIndex[1], startNodeIndex[2], param.length, param.dx);
        int startNodePointIndex = (*pointIndexFind)[startNodeID];

        Eigen::Vector3i stopNodeIndex = (*boundaryNodesPosIndex)[stopNode];
        if (ifCriticalNode(stopNodeIndex.cast<double>() * param.dx, param, criticalNodeIndex) == true) {
            stopNodeIndex = findNearestBoundaryNode(stopNode, points, boundaryNodesID, boundaryNodesPosIndex, pointIndexFind, param, criticalNodeIndex);
            // if this point is a desolate critical point, return true. Keep this face though it may become a tooth.
            if (stopNodeIndex[0] < 0) {
                return true;
            }
        }
        int stopNodeID = calculateID(stopNodeIndex[0], stopNodeIndex[1], stopNodeIndex[2], param.length, param.dx);
        int stopNodePointIndex = (*pointIndexFind)[stopNodeID];

        //cout << "Real start and stop points are: " << endl;
        //cout << "startNodeIndex = "<< startNodePointIndex <<" pos "<< startNodeIndex[0] << " " << startNodeIndex[1] << " " << startNodeIndex[2] << " " << endl;
        //cout << "stopNodeIndex = " << stopNodePointIndex << " pos " << stopNodeIndex[0] << " " << stopNodeIndex[1] << " " << stopNodeIndex[2] << " " << endl<<endl;

        std::vector<int> sameSideNeighbours; // the same side neighbours
        std::vector<int> otherSideNeighbours; // the other side neighbours

        // find initial cube bounding neighbours of startNode
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    if (i + j + k != 0) {
                        Eigen::Vector3i normal = { i, j, k };
                        Eigen::Vector3i neighbourNodePosIndex = startNodeIndex + normal;

                        int neighbourNodeID = calculateID(neighbourNodePosIndex[0], neighbourNodePosIndex[1], neighbourNodePosIndex[2], param.length, param.dx);

                        if (count((*boundaryNodesID).begin(), (*boundaryNodesID).end(), neighbourNodeID) != 0) {
                            int neighbourNodePointIndex = (*pointIndexFind)[neighbourNodeID];
                            sameSideNeighbours.push_back(neighbourNodePointIndex);
                        }
                    }
                }
            }
        }

        // find the same side neighbours of this point
        for (int f = 0; f < (*points)[startNodePointIndex].neighbourSameSide.size(); f++) {
            int sameSideNeighbourPoint = (*points)[startNodePointIndex].neighbourSameSide[f];
            if (count(sameSideNeighbours.begin(), sameSideNeighbours.end(), sameSideNeighbourPoint) == 0) {
                sameSideNeighbours.push_back(sameSideNeighbourPoint);
            }
        }

        // find the other side neighbours of this point
        for (int f = 0; f < (*points)[startNodePointIndex].neighbourOtherSide.size(); f++) {
            int otherSideNeighbourPoint = (*points)[startNodePointIndex].neighbourOtherSide[f];
            if (count(otherSideNeighbours.begin(), otherSideNeighbours.end(), otherSideNeighbourPoint) == 0) {
                otherSideNeighbours.push_back(otherSideNeighbourPoint);
            }
        }

        //cout << "Initial same side neighbours are: ";
        //for (int i = 0; i < sameSideNeighbours.size(); i++)
        //{
        //	cout << sameSideNeighbours[i] << " ";
        //}

        //cout <<endl<< "Initial the other side neighbours are: ";
        //for (int i = 0; i < otherSideNeighbours.size(); i++)
        //{
        //	cout << otherSideNeighbours[i] << " ";
        //}
        //cout << endl << endl << endl << endl;

        bool findStopNode = false;
        int sameSideALayerStart = 0, sameSideALayerLength = sameSideNeighbours.size();
        int otherSideALayerStart = 0, otherSideALayerLength = otherSideNeighbours.size();
        do {

            int sameSideNextLayerLength = 0;
            int otherSideNextLayerLength = 0;

            for (int s = sameSideALayerStart; s < sameSideALayerStart + sameSideALayerLength; s++) {
                int nodePointIndex = sameSideNeighbours[s];

                // find the same side neighbours
                for (int k = 0; k < (*points)[nodePointIndex].neighbourSameSide.size(); k++) {
                    int sameSideNeighbourID = (*points)[nodePointIndex].neighbourSameSide[k];

                    if (sameSideNeighbourID == stopNodePointIndex) {
                        return false;
                    }
                    if (count(sameSideNeighbours.begin(), sameSideNeighbours.end(), sameSideNeighbourID) == 0) {
                        sameSideNeighbours.push_back(sameSideNeighbourID);
                        sameSideNextLayerLength += 1;
                    }
                }

                // find the other side neighbours
                for (int m = 0; m < (*points)[nodePointIndex].neighbourOtherSide.size(); m++) {
                    int otherSideNeighbourID = (*points)[nodePointIndex].neighbourOtherSide[m];

                    if (otherSideNeighbourID == stopNodePointIndex) {

                        return true;
                    }
                    if (count(otherSideNeighbours.begin(), otherSideNeighbours.end(), otherSideNeighbourID) == 0) {
                        //cout << "nodePointIndex = " << nodePointIndex << "; otherSideNeighbourID = " << otherSideNeighbourID << endl;
                        otherSideNeighbours.push_back(otherSideNeighbourID);
                        //otherSideNextLayerLength += 1;
                        otherSideALayerLength += 1;
                    }
                }
            }

            for (int s = otherSideALayerStart; s < otherSideALayerStart + otherSideALayerLength; s++) {

                int nodePointIndex = otherSideNeighbours[s];
                // find the same side neighbours; "The same" means in the other crack side
                for (int k = 0; k < (*points)[nodePointIndex].neighbourSameSide.size(); k++) {
                    int sameSideNeighbourID = (*points)[nodePointIndex].neighbourSameSide[k];
                    if (sameSideNeighbourID == stopNodePointIndex) {
                        return true;
                    }
                    if (count(otherSideNeighbours.begin(), otherSideNeighbours.end(), sameSideNeighbourID) == 0) {
                        otherSideNeighbours.push_back(sameSideNeighbourID);
                        otherSideNextLayerLength += 1;
                    }
                }

                //// find the other side neighbours; "The other" means in the same side of startNode
                //for (int m = 0; m < (*points)[nodePointIndex].neighbourOtherSide.size(); m++)
                //{
                //	int otherSideNeighbourID = (*points)[nodePointIndex].neighbourOtherSide[m];
                //	if (otherSideNeighbourID == stopNodePointIndex)
                //	{
                //		return false;
                //	}
                //	if (count(sameSideNeighbours.begin(), sameSideNeighbours.end(), otherSideNeighbourID) == 0)
                //	{
                //		sameSideNeighbours.push_back(otherSideNeighbourID);
                //		sameSideNextLayerLength += 1;
                //	}

                //}
            }

            sameSideALayerStart = sameSideALayerStart + sameSideALayerLength;
            sameSideALayerLength = sameSideNextLayerLength;
            otherSideALayerStart = otherSideALayerStart + otherSideALayerLength;
            otherSideALayerLength = otherSideNextLayerLength;

            if (sameSideNextLayerLength + sameSideNextLayerLength == 0) {
                return true;
            }

        } while (findStopNode == false);
    }

    // Extract the crack surface
    static std::tuple<bool, meshObjFormat, meshObjFormat, std::vector<meshObjFormat>> extractCrackSurf(std::vector<Particle>* particlesRaw, struct parametersSim param, int timestep)
    {

        std::cout << "Start extracting" << std::endl;

        //*********Read fully damaged particles***********//
        std::vector<Particle> fullyDamagedParticles;
        std::map<int, int> fullyDamagedParticlesGridMap;
        std::vector<Grid> fullyDamagedParticlesNodesVec;

        readParticles(particlesRaw, &fullyDamagedParticles, true, param);
        calDamageGradient(&fullyDamagedParticles, param, param.dx, &fullyDamagedParticlesGridMap, &fullyDamagedParticlesNodesVec);
        setNodeValue(&fullyDamagedParticlesNodesVec, 1);
        findBoundaryNodes(&fullyDamagedParticles, &fullyDamagedParticlesNodesVec, &fullyDamagedParticlesGridMap, param, 2); // This bounding box is used to generate boundary nodes
        findBoundaryNodes(&fullyDamagedParticles, &fullyDamagedParticlesNodesVec, &fullyDamagedParticlesGridMap, param, 3); // This bounding box is used to generate clip surface mesh
        //*********Read fully damaged particles***********//

        //*********Read all particles***********//
        std::vector<Particle> allParticles;
        std::map<int, int> allParticlesGridMap;
        std::vector<Grid> allParticlesNodesVec;

        readParticles(particlesRaw, &allParticles, false, param);
        calDamageGradient(&allParticles, param, param.dx, &allParticlesGridMap, &allParticlesNodesVec);
        setNodeValue(&allParticlesNodesVec, 1);
        findBoundaryNodes(&allParticles, &allParticlesNodesVec, &allParticlesGridMap, param, 2); // This bounding box is used to generate boundary nodes
        findBoundaryNodes(&allParticles, &allParticlesNodesVec, &allParticlesGridMap, param, 3); // This bounding box is used to generate clip surface mesh
        //*********Read all particles**********

        // find boundary nodes
        std::vector<int> allParticlesNodeBoundaryIndex; // store the index or ID of allParticles boundary nodes
        for (int i = 0; i < allParticlesNodesVec.size(); i++) {
            if (allParticlesNodesVec[i].Di == 2) {
                int ID = calculateID(allParticlesNodesVec[i].posIndex[0], allParticlesNodesVec[i].posIndex[1], allParticlesNodesVec[i].posIndex[2], param.length, param.dx);
                allParticlesNodeBoundaryIndex.push_back(ID);
            }
        }

        //************find surface nodes//
        std::vector<int> surfaceNodesID; // store surface nodes ID
        std::vector<Eigen::Vector3i> surfaceNodesPosIndex; // store surface position index
        std::vector<int> boundaryNodesIDNoClean; // store boundary nodes ID
        std::vector<Eigen::Vector3i> boundaryNodesPosIndexNoClean; // store boundary position index
        for (int i = 0; i < fullyDamagedParticlesNodesVec.size(); i++) {
            if (fullyDamagedParticlesNodesVec[i].Di == 2) {
                int ID = calculateID(fullyDamagedParticlesNodesVec[i].posIndex[0], fullyDamagedParticlesNodesVec[i].posIndex[1], fullyDamagedParticlesNodesVec[i].posIndex[2], param.length, param.dx);
                if (count(allParticlesNodeBoundaryIndex.begin(), allParticlesNodeBoundaryIndex.end(), ID) == 0) {
                    boundaryNodesPosIndexNoClean.push_back(fullyDamagedParticlesNodesVec[i].posIndex);
                    boundaryNodesIDNoClean.push_back(ID);
                }

                //************find surface nodes//
                if (count(allParticlesNodeBoundaryIndex.begin(), allParticlesNodeBoundaryIndex.end(), ID) != 0) {
                    int surfaceNodeID = calculateID(fullyDamagedParticlesNodesVec[i].posIndex[0], fullyDamagedParticlesNodesVec[i].posIndex[1], fullyDamagedParticlesNodesVec[i].posIndex[2], param.length, param.dx);
                    surfaceNodesID.push_back(surfaceNodeID);
                    surfaceNodesPosIndex.push_back(fullyDamagedParticlesNodesVec[i].posIndex);
                }
                //************find surface nodes//
            }
        }

        // clean isolate nodes and sharp nodes
        std::vector<int> boundaryNodesID; // store boundary nodes ID
        std::vector<Eigen::Vector3i> boundaryNodesPosIndex; // store boundary position index
        for (int m = 0; m < boundaryNodesPosIndexNoClean.size(); m++) {
            bool stopIsolate = false;
            for (int axis = 0; axis < 3; axis++) // three axis directions
            {
                if (stopIsolate == false) {
                    for (int pn = 0; pn < 2; pn++) {
                        if (stopIsolate == false) {
                            Eigen::Vector3i normal = { 0, 0, 0 };
                            normal[axis] = 2 * pn - 1;
                            Eigen::Vector3i nodePosIndex = boundaryNodesPosIndexNoClean[m] + normal;
                            int nodeID = calculateID(nodePosIndex[0], nodePosIndex[1], nodePosIndex[2], param.length, param.dx);
                            if (count(boundaryNodesIDNoClean.begin(), boundaryNodesIDNoClean.end(), nodeID) != 0) {
                                boundaryNodesID.push_back(boundaryNodesIDNoClean[m]);
                                boundaryNodesPosIndex.push_back(boundaryNodesPosIndexNoClean[m]);
                                stopIsolate = true;
                            }
                        }
                    }
                }
            }
        }

        // define a std::map that can find the index of a point in the point vector
        std::map<int, int> pointIndexFind;
        for (int m = 0; m < boundaryNodesID.size(); m++) {
            pointIndexFind[boundaryNodesID[m]] = m;
        }




        std::cout << "The number of boundary nodes is " << boundaryNodesID.size() << std::endl;

        // find critical nodes
        std::vector<int> criticalNodeIndex;
        int criticalNodeVolumeLength = 1;
        for (int m = 0; m < surfaceNodesPosIndex.size(); m++) {

            for (int i = -criticalNodeVolumeLength; i < criticalNodeVolumeLength + 1; i++) {
                for (int j = -criticalNodeVolumeLength; j < criticalNodeVolumeLength + 1; j++) {
                    for (int k = -criticalNodeVolumeLength; k < criticalNodeVolumeLength + 1; k++) {
                        Eigen::Vector3i increment = { i, j, k };
                        Eigen::Vector3i nodePosIndex = surfaceNodesPosIndex[m] + increment;
                        int ID = calculateID(nodePosIndex[0], nodePosIndex[1], nodePosIndex[2], param.length, param.dx);
                        if (count(boundaryNodesID.begin(), boundaryNodesID.end(), ID) != 0) {
                            if (count(criticalNodeIndex.begin(), criticalNodeIndex.end(), ID) == 0) {
                                criticalNodeIndex.push_back(ID);
                            }
                        }
                    }
                }
            }
        }

        std::cout << "Start voro++" << std::endl;

        double x_min = 1.0E10, x_max = -1.0E10;
        double y_min = 1.0E10, y_max = -1.0E10;
        double z_min = 1.0E10, z_max = -1.0E10;
        int n_x = 10, n_y = 10, n_z = 10;

        for (int m = 0; m < (*particlesRaw).size(); m++) {
            x_min = std::min(x_min, (*particlesRaw)[m].pos[0]);
            y_min = std::min(y_min, (*particlesRaw)[m].pos[1]);
            z_min = std::min(z_min, (*particlesRaw)[m].pos[2]);

            x_max = std::max(x_max, (*particlesRaw)[m].pos[0]);
            y_max = std::max(y_max, (*particlesRaw)[m].pos[1]);
            z_max = std::max(z_max, (*particlesRaw)[m].pos[2]);
        }
        x_min = x_min - 4 * param.dx;
        x_max = x_max + 4 * param.dx;
        y_min = y_min - 4 * param.dx;
        y_max = y_max + 4 * param.dx;
        z_min = z_min - 4 * param.dx;
        z_max = z_max + 4 * param.dx;

        voro::pre_container pcon(x_min, x_max, y_min, y_max, z_min, z_max, false, false, false);
        // Import the particles from a file
        for (int m = 0; m < boundaryNodesPosIndex.size(); m++)
        {
            pcon.put(m, boundaryNodesPosIndex[m][0] * param.dx, boundaryNodesPosIndex[m][1] * param.dx, boundaryNodesPosIndex[m][2] * param.dx);
        }
        pcon.guess_optimal(n_x, n_y, n_z);
        voro::container con(x_min, x_max, y_min, y_max, z_min, z_max, n_x, n_y, n_z, false, false, false, 8);
        pcon.setup(con);

        std::vector<Point> points;
        for (int i = 0; i < boundaryNodesPosIndex.size(); i++) {
            Eigen::Vector3d pos = { 0, 0, 0 }; // each point's position
            std::vector<Eigen::Vector3d> verticsCoor; // vertices' coordinate
            std::vector<std::vector<int>> verticesFace; // vertices of each face
            std::vector<Eigen::Vector3d> surfaceNormal; // vertices' coordinate
            std::vector<int> neighbour; // neighbour points that share common faces with this point
            std::vector<int> neighbourCalculated; // neighbour points that have already find shared face
            points.push_back(Point(-999, pos, 0, verticsCoor, 0, verticesFace, surfaceNormal, neighbour));
        }

        if (points.size() == 0)
        {
            bool findCrackSurface = false;
            meshObjFormat crackSurfacePartialCut;
            meshObjFormat crackSurfaceFullCut;
            std::vector<meshObjFormat> allFragmentsObj;
            std::tuple<bool, meshObjFormat, meshObjFormat, std::vector<meshObjFormat>> resultReturn(findCrackSurface, crackSurfacePartialCut, crackSurfaceFullCut, allFragmentsObj);
            return resultReturn;
        }


        voro::c_loop_all cl(con);
        if (cl.start()) {

            double x, y, z;
            int index;
            voro::voronoicell_neighbor c;
            std::vector<int> neighbour, verticesFace;
            std::vector<double> vertices;
            std::vector<double> normals;
            do
                if (con.compute_cell(c, cl)) {

                    int index = cl.pid();
                    points[index].index = index;

                    cl.pos(x, y, z);
                    points[index].pos = { x, y, z };

                    c.vertices(x, y, z, vertices);
                    points[index].numVertices = vertices.size() / 3;
                    for (int m = 0; m < vertices.size() / 3; m++) {
                        Eigen::Vector3d vert = { vertices[m * 3], vertices[m * 3 + 1], vertices[m * 3 + 2] };
                        points[index].verticsCoor.push_back(vert);
                        points[index].globalIndexed.push_back(-99);
                    }

                    c.neighbors(neighbour);
                    points[index].neighbour = neighbour;
                    points[index].numFaces = neighbour.size();

                    c.face_vertices(verticesFace);
                    int start = 1, end = verticesFace[0] + 1;
                    do {
                        std::vector<int> faceVert;
                        for (int m = start; m < end; m++) {
                            faceVert.push_back(verticesFace[m]);
                        }
                        points[index].verticesFace.push_back(faceVert);
                        points[index].faceIndexed.push_back(false);
                        start = end + 1;
                        end += verticesFace[end] + 1;
                    } while (points[index].verticesFace.size() != neighbour.size());

                    c.normals(normals);
                    for (int m = 0; m < normals.size() / 3; m++) {
                        Eigen::Vector3d normal = { normals[m * 3], normals[m * 3 + 1], normals[m * 3 + 2] };
                        points[index].surfaceNormal.push_back(normal);
                    }
                }
            while (cl.inc());
        }

        std::cout << "Voro++ finished" << std::endl;


        // Find global vertex of each cell
        std::vector<Eigen::Vector3d> globalPoints;
        {
            for (int i = 0; i < points.size(); i++)
            {

                for (int k = 0; k < points[i].neighbour.size(); k++)
                {
                    int neighbourIndex = points[i].neighbour[k];
                    if (neighbourIndex > 0) // remove bounding box faces
                    {
                        if (points[i].faceIndexed[k] == false)
                        {

                            points[i].faceIndexed[k] = true;
                            std::vector<int> curr_face = points[i].verticesFace[k];
                            std::vector<int> neig_face;
                            // find neighbour cell's face that corresponds to this cell's face
                            for (int m = 0; m < points[neighbourIndex].neighbour.size(); m++)
                            {
                                if (points[neighbourIndex].neighbour[m] == i)// the neighbour cell's m face corresponds current cell's k face
                                {
                                    neig_face = points[neighbourIndex].verticesFace[m];
                                    points[neighbourIndex].faceIndexed[m] = true;
                                    break;
                                }
                            }
                            std::reverse(neig_face.begin(), neig_face.end());

                            // !!!!!!! Voro++ is unstable. Here is a failure case: cell_j has no corresponding face wrt cell_i even if there are neighbours
                            if (neig_face.size() != curr_face.size())
                            {
                                std::cout << "!!!! Voro++ internal error: No faces correspondence!"  << std::endl;
                                std::vector<meshObjFormat> allFragmentsObj;
                                meshObjFormat crackSurfacePartialCut, crackSurfaceFullCut;
                                std::tuple<bool, meshObjFormat, meshObjFormat, std::vector<meshObjFormat>> resultReturn(false, crackSurfacePartialCut, crackSurfaceFullCut, allFragmentsObj);
                                return resultReturn;
                            }

                            // extend the vector to facilite indexing
                            std::vector<int> neig_face_ext = neig_face;
                            for (int sd = 0; sd < neig_face.size(); sd++)
                            {
                                neig_face_ext.push_back(neig_face[sd]);
                            }

                            double sumup_max = 1.0E7; // maximum sumup
                            int correspondVert = -99;
                            for (int sd = 0; sd < curr_face.size(); sd++)
                            {
                                double sumup = 0;
                                for (int wd = sd; wd < sd + neig_face.size(); wd++)
                                {

                                    sumup += (points[i].verticsCoor[curr_face[wd - sd]] - points[neighbourIndex].verticsCoor[neig_face_ext[wd]]).norm();
                                }
                                if (sumup < sumup_max)
                                {
                                    correspondVert = sd;
                                    sumup_max = sumup;
                                }
                            }

                            // the final one-to-one correspondance of two cell's faces
                            for (int sd = 0; sd < curr_face.size(); sd++)
                            {
                                int curr_ver = curr_face[sd];
                                int neig_ver = neig_face_ext[sd + correspondVert];
                                //std::cout << sd << ", " << curr_ver << ", " << neig_ver << std::endl;
                                if (points[i].globalIndexed[curr_ver] == -99)
                                {
                                    if (points[neighbourIndex].globalIndexed[neig_ver] == -99)
                                    {
                                        int ptInd = globalPoints.size();
                                        points[i].globalIndexed[curr_ver] = ptInd;
                                        points[neighbourIndex].globalIndexed[neig_ver] = ptInd;

                                        globalPoints.push_back(points[i].verticsCoor[curr_ver]);
                                    }
                                    else
                                    {
                                        int ptInd = points[neighbourIndex].globalIndexed[neig_ver];
                                        points[i].globalIndexed[curr_ver] = ptInd;
                                    }
                                }
                                else
                                {
                                    if (points[neighbourIndex].globalIndexed[neig_ver] == -99)
                                    {
                                        int ptInd = points[i].globalIndexed[curr_ver];
                                        points[neighbourIndex].globalIndexed[neig_ver] = ptInd;
                                    }
                                }


                            }








                        }

                    }
                    else
                    {

                    }


                }
            }


        }




        //***********Read all particles and calculate the damage gradient*************//
        std::vector<Particle> particles;
        std::map<int, int> gridMap;
        std::vector<Grid> nodesVec;
        readParticlesAndCalGradient(&fullyDamagedParticlesNodesVec, &particles, param, &gridMap, &nodesVec);

        double pi = 3.141592653;
        double radius = param.dx * sqrt(3); // support radius of two points
        //radius = 0;

        std::vector<Eigen::Vector3d> verticesTmp; // vertex index
        std::vector<std::vector<int>> facesTmp;

        std::cout << "Start extracting interior faces" << std::endl;
        bool newAlgoRemoveDup = true;

        std::vector<std::set<int>> sharedFace(points.size()); // store the point index which shares a crack surface with the other point

        // find faces that are in the interior and store neighbour information
        for (int i = 0; i < points.size(); i++) {
            Eigen::Vector3d pos = points[i].pos;
            if (ifCriticalNode(pos, param, &criticalNodeIndex) == false) {
                for (int k = 0; k < points[i].neighbour.size(); k++) {
                    int neighbourIndex = points[i].neighbour[k];
                    if (neighbourIndex > 0) // remove bounding box faces
                    {

                        Eigen::Vector3d posNeig = points[neighbourIndex].pos;
                        Eigen::Vector3d posDiff = pos - posNeig;
                        double distancePair = posDiff.norm();

                        if (ifCriticalNode(posNeig, param, &criticalNodeIndex) == false) {
                            if (count(points[i].neighbourCalculated.begin(), points[i].neighbourCalculated.end(), neighbourIndex) == 0) // if these two neighbours have not yet been compared
                            {

                                if (distancePair > radius) // if their distance is larger than the threshold
                                {
                                    double existInCrack = ifFullyDamaged((pos + posNeig) / 2.0, param, &gridMap, &nodesVec);
                                    if (existInCrack >= 1.0) // if the middle point is located in the crack area
                                    {

                                        std::vector<int> faceVerteice;
                                        for (int ver = 0; ver < points[i].verticesFace[k].size(); ver++) {
                                            int indexVertex = points[i].verticesFace[k][ver];
                                            Eigen::Vector3d vertex = points[i].verticsCoor[indexVertex];
                                            int vertexIndex = findIndexVertex(vertex, &verticesTmp);

                                            if (newAlgoRemoveDup)
                                            {
                                                int vertGlobalIndex = points[i].globalIndexed[indexVertex];
                                                faceVerteice.push_back(vertGlobalIndex);
                                            }
                                            else
                                            {
                                                faceVerteice.push_back(vertexIndex);
                                            }

                                        }
                                        facesTmp.push_back(faceVerteice);

                                        sharedFace[i].insert(neighbourIndex);
                                        sharedFace[neighbourIndex].insert(i);

                                        points[i].neighbourOtherSide.push_back(neighbourIndex);
                                        points[neighbourIndex].neighbourOtherSide.push_back(i);

                                    }
                                    else {
                                        points[i].neighbourSameSide.push_back(neighbourIndex);
                                        points[neighbourIndex].neighbourSameSide.push_back(i);
                                    }

                                    // store computation information.
                                    points[i].neighbourCalculated.push_back(neighbourIndex);
                                    points[neighbourIndex].neighbourCalculated.push_back(i);

                                }
                                else {
                                    points[i].neighbourSameSide.push_back(neighbourIndex);
                                    points[neighbourIndex].neighbourSameSide.push_back(i);

                                    // store computation information.
                                    points[i].neighbourCalculated.push_back(neighbourIndex);
                                    points[neighbourIndex].neighbourCalculated.push_back(i);
                                }
                            }
                        }
                    }
                }
            }
        }

        std::cout << "Start extracting boundary faces" << std::endl;
        // find faces that are in the interior and store neighbour information
        for (int i = 0; i < points.size(); i++) {
            Eigen::Vector3d pos = points[i].pos;

            if (ifCriticalNode(pos, param, &criticalNodeIndex) == false) {
                for (int k = 0; k < points[i].neighbour.size(); k++) {
                    int neighbourIndex = points[i].neighbour[k];
                    if (neighbourIndex > 0) // remove bounding box faces
                    {

                        Eigen::Vector3d posNeig = points[neighbourIndex].pos;
                        Eigen::Vector3d posDiff = pos - posNeig;
                        double distancePair = posDiff.norm();

                        if (ifCriticalNode(posNeig, param, &criticalNodeIndex) == true) {
                            if (count(points[i].neighbourCalculated.begin(), points[i].neighbourCalculated.end(), neighbourIndex) == 0) // if these two neighbours have not yet been compared
                            {

                                if (distancePair > radius) // if their distance is larger than the threshold
                                {
                                    bool twoSide1 = ifTwoSides(i, neighbourIndex, &points, &boundaryNodesID, &boundaryNodesPosIndex, &pointIndexFind, param, &criticalNodeIndex);

                                    //twoSide1 = true;
                                    if (twoSide1 == true) {
                                        std::vector<int> faceVerteice;
                                        for (int ver = 0; ver < points[i].verticesFace[k].size(); ver++) {
                                            int indexVertex = points[i].verticesFace[k][ver];
                                            Eigen::Vector3d vertex = points[i].verticsCoor[indexVertex];
                                            int vertexIndex = findIndexVertex(vertex, &verticesTmp);

                                            if (newAlgoRemoveDup)
                                            {
                                                int vertGlobalIndex = points[i].globalIndexed[indexVertex];
                                                faceVerteice.push_back(vertGlobalIndex);
                                            }
                                            else
                                            {
                                                faceVerteice.push_back(vertexIndex);
                                            }
                                        }
                                        facesTmp.push_back(faceVerteice);

                                        sharedFace[i].insert(neighbourIndex);
                                        sharedFace[neighbourIndex].insert(i);

                                        points[i].neighbourOtherSide.push_back(neighbourIndex);
                                        points[neighbourIndex].neighbourOtherSide.push_back(i);
                                    }
                                    else {
                                        points[i].neighbourSameSide.push_back(neighbourIndex);
                                        points[neighbourIndex].neighbourSameSide.push_back(i);
                                    }

                                    // store computation information.
                                    points[i].neighbourCalculated.push_back(neighbourIndex);
                                    points[neighbourIndex].neighbourCalculated.push_back(i);

                                }
                                else {
                                    points[i].neighbourSameSide.push_back(neighbourIndex);
                                    points[neighbourIndex].neighbourSameSide.push_back(i);

                                    // store computation information.
                                    points[i].neighbourCalculated.push_back(neighbourIndex);
                                    points[neighbourIndex].neighbourCalculated.push_back(i);
                                }
                            }
                        }
                    }
                }
            }
        }

        // find faces that defined by critical point
        for (int i = 0; i < points.size(); i++) {

            Eigen::Vector3d pos = points[i].pos;
            if (ifCriticalNode(pos, param, &criticalNodeIndex) == true) {

                for (int k = 0; k < points[i].neighbour.size(); k++) {
                    int neighbourIndex = points[i].neighbour[k];
                    if (neighbourIndex > 0) // remove bounding box faces
                    {
                        Eigen::Vector3d posNeig = points[neighbourIndex].pos;
                        Eigen::Vector3d posDiff = pos - posNeig;
                        double distancePair = posDiff.norm();

                        if (count(points[i].neighbourCalculated.begin(), points[i].neighbourCalculated.end(), neighbourIndex) == 0) // if these two neighbours have not yet been compared
                        {

                            if (distancePair > radius) // if their distance is larger than the threshold
                            {

                                bool twoSide1 = ifTwoSides(i, neighbourIndex, &points, &boundaryNodesID, &boundaryNodesPosIndex, &pointIndexFind, param, &criticalNodeIndex);

                                if (twoSide1 == true) {

                                    std::vector<int> faceVerteice;
                                    for (int ver = 0; ver < points[i].verticesFace[k].size(); ver++) {
                                        int indexVertex = points[i].verticesFace[k][ver];
                                        Eigen::Vector3d vertex = points[i].verticsCoor[indexVertex];
                                        int vertexIndex = findIndexVertex(vertex, &verticesTmp);

                                        if (newAlgoRemoveDup)
                                        {
                                            int vertGlobalIndex = points[i].globalIndexed[indexVertex];
                                            faceVerteice.push_back(vertGlobalIndex);
                                        }
                                        else
                                        {
                                            faceVerteice.push_back(vertexIndex);
                                        }
                                    }
                                    facesTmp.push_back(faceVerteice);

                                    sharedFace[i].insert(neighbourIndex);
                                    sharedFace[neighbourIndex].insert(i);
                                }

                                // store computation information.
                                points[i].neighbourCalculated.push_back(neighbourIndex);
                                points[neighbourIndex].neighbourCalculated.push_back(i);
                            }
                        }
                    }
                }
            }
        }





        ////////////////////////////////
        // calculate all fragments
        ////////////////////////////////
        std::set<int> remainingPoints;
        for (int i = 0; i < points.size(); i++) {
            remainingPoints.insert(i);
        }

        std::vector<std::vector<int>> allFragments;

        do {

            std::vector<int> fragment;
            int firstElement = *remainingPoints.begin();
            fragment.push_back(firstElement);
            remainingPoints.erase(firstElement);
            int start = 0; // the index of starting pointof a single layer
            int countLayer = 1;
            do {
                int countLayerUpdate = 0;
                for (int i = start; i < start + countLayer; i++) // parse each candidate
                {
                    int currentPoint = fragment[i];
                    for (int j = 0; j < points[currentPoint].neighbour.size(); j++) {
                        int candPoint = points[currentPoint].neighbour[j];
                        if (candPoint >= 0) {
                            if (sharedFace[currentPoint].find(candPoint) == sharedFace[currentPoint].end()) {
                                if (std::find(fragment.begin(), fragment.end(), candPoint) == fragment.end()) {
                                    fragment.push_back(candPoint);
                                    remainingPoints.erase(candPoint);
                                    countLayerUpdate += 1;
                                }
                            }
                        }
                    }
                }
                start = start + countLayer;
                countLayer = countLayerUpdate;

            } while (countLayer != 0);

            allFragments.push_back(fragment);

        } while (remainingPoints.size() != 0);

        std::cout << "Number of raw fragments is =" << allFragments.size() << std::endl;

        // add interior volume to the surrounding fragment
        std::vector<std::vector<int>> allFragmentsRemoveInterior; // store each fragment where interior volume is added to surrounding volume
        std::set<int> interiorFragmentIndex; // store the index of fragment which is an interior volume
        for (int i = 0; i < allFragments.size(); i++) {
            std::vector<int> fragment = allFragments[i];
            bool interior = true;
            for (int j = 0; j < fragment.size(); j++) {
                for (int h = 0; h < points[fragment[j]].neighbour.size(); h++) {
                    int neigPoint = points[fragment[j]].neighbour[h];
                    if (neigPoint < 0) {
                        interior = false;
                        break;
                    }
                }

                if (interior == false) {
                    break;
                }
            }

            if (interior == true) {
                interiorFragmentIndex.insert(i);
            }
        }

        if (interiorFragmentIndex.size() == 0) // all fargments are not interior
        {
            allFragmentsRemoveInterior = allFragments;
        }
        else {
            std::map<int, int> mergeInterior; // key and value are the index of interior volume and surrounding volume respectively

            // find the surrounding volume that an interior volume should be added into
            std::set<int>::iterator it;
            for (it = interiorFragmentIndex.begin(); it != interiorFragmentIndex.end(); ++it) {
                // find neighbour point in the surrounding volume
                int surroundPoint = -99; // surrounding neighbour point of this interior volume
                for (int k = 0; k < allFragments[*it].size(); k++) {
                    int seedPoint = allFragments[*it][k];
                    for (int i = 0; i < points[seedPoint].neighbour.size(); i++) {
                        int neigPoint = points[seedPoint].neighbour[i];
                        if (sharedFace[seedPoint].find(neigPoint) != sharedFace[seedPoint].end()) {
                            surroundPoint = neigPoint;
                            break;
                        }
                    }

                    if (surroundPoint >= 0) {
                        break;
                    }
                }

                // find surrounding volume
                int surroundingVolume = -99;
                for (int i = 0; i < allFragments.size(); i++) {
                    if (std::find(allFragments[i].begin(), allFragments[i].end(), surroundPoint) != allFragments[i].end()) {
                        surroundingVolume = i;
                        break;
                    }
                }
                mergeInterior[*it] = surroundingVolume;
                if (surroundingVolume < 0) {
                    bool findCrackSurface = false;
                    meshObjFormat nocrackmesh;
                    std::vector<meshObjFormat> noFragments;
                    std::tuple<bool, meshObjFormat, meshObjFormat, std::vector<meshObjFormat>> resultReturn(findCrackSurface, nocrackmesh, nocrackmesh, noFragments);
                    return resultReturn;
                }
            }

            // add interior volume into the surrounding volume
            std::map<int, int>::iterator itMap;
            for (itMap = mergeInterior.begin(); itMap != mergeInterior.end(); itMap++) {
                int interiorFragment = itMap->first;
                int surroundingFragment = itMap->second;
                allFragments[surroundingFragment].insert(allFragments[surroundingFragment].end(), allFragments[interiorFragment].begin(), allFragments[interiorFragment].end());
            }

            for (int i = 0; i < allFragments.size(); i++) {
                if (mergeInterior.find(i) == mergeInterior.end()) {
                    allFragmentsRemoveInterior.push_back(allFragments[i]);
                }
            }
        }

        std::cout << "Number of final fragments is =" << allFragmentsRemoveInterior.size() << std::endl;

        // remove duplicated vertices
        std::vector<meshObjFormat> allFragmentsObj;
        for (int i = 0; i < allFragmentsRemoveInterior.size(); i++) {
            std::vector<int> fragment = allFragmentsRemoveInterior[i];

            std::vector<Eigen::Vector3d> verticesEachFrag;
            std::vector<std::vector<int>> facesEachFrag;
            for (int k = 0; k < fragment.size(); k++) {
                // find faces of each voronoi cell
                std::set<int> voroCellVertices;
                std::vector<std::vector<int>> voroCellFaces;
                for (int h = 0; h < points[fragment[k]].verticesFace.size(); h++) {
                    int opponentPoint = points[fragment[k]].neighbour[h];
                    if (std::find(fragment.begin(), fragment.end(), opponentPoint) == fragment.end()) {
                        voroCellFaces.push_back(points[fragment[k]].verticesFace[h]);
                        for (int f = 0; f < points[fragment[k]].verticesFace[h].size(); f++) {
                            voroCellVertices.insert(points[fragment[k]].verticesFace[h][f]);
                        }
                    }
                }

                // remove duplicated vertices
                std::map<int, int> verticesMapping;
                std::set<int>::iterator it;
                for (it = voroCellVertices.begin(); it != voroCellVertices.end(); ++it) {
                    Eigen::Vector3d candiVert = points[fragment[k]].verticsCoor[*it];

                    if (verticesEachFrag.size() == 0) {
                        verticesMapping[*it] = 0;
                        verticesEachFrag.push_back(candiVert);
                    }
                    else {
                        // remove duplicated vertices
                        int veryIndex = -1;
                        bool findDup = false;
                        do {
                            veryIndex += 1;
                            Eigen::Vector3d existVert = verticesEachFrag[veryIndex];
                            if (candiVert[0] == existVert[0] && candiVert[1] == existVert[1] && candiVert[2] == existVert[2]) {
                                findDup = true;
                            }
                        } while (findDup == false && veryIndex != verticesEachFrag.size() - 1);

                        if (findDup == true) {
                            verticesMapping[*it] = veryIndex;
                        }
                        else {
                            verticesMapping[*it] = verticesEachFrag.size();
                            verticesEachFrag.push_back(candiVert);
                        }
                    }
                }

                // reorder face vertices
                for (int f = 0; f < voroCellFaces.size(); f++) {
                    std::vector<int> orderedFace;
                    for (int d = 0; d < voroCellFaces[f].size(); d++) {
                        orderedFace.push_back(verticesMapping[voroCellFaces[f][d]]);
                    }
                    facesEachFrag.push_back(orderedFace);
                }
            }

            meshObjFormat fragmentObj;
            fragmentObj.faces = facesEachFrag;
            fragmentObj.vertices = verticesEachFrag;
            allFragmentsObj.push_back(fragmentObj);
        }



        // find crack surface that does fully cut
        std::vector<Eigen::Vector3d> vertices;
        std::vector<std::vector<int>> faces;
        std::set<std::string> faceSampled;
        for (int i = 0; i < allFragmentsRemoveInterior.size(); i++) {
            std::vector<int> fragment = allFragmentsRemoveInterior[i];

            for (int k = 0; k < fragment.size(); k++) {
                // find faces of each voronoi cell
                for (int h = 0; h < points[fragment[k]].verticesFace.size(); h++) {
                    int opponentPoint = points[fragment[k]].neighbour[h];
                    if (opponentPoint >= 0 && std::find(fragment.begin(), fragment.end(), opponentPoint) == fragment.end()) {
                        std::string facePositive = std::to_string(fragment[k]) + "#" + std::to_string(opponentPoint);
                        std::string faceNegative = std::to_string(opponentPoint) + "#" + std::to_string(fragment[k]);
                        if (faceSampled.find(facePositive) == faceSampled.end() || faceSampled.find(faceNegative) == faceSampled.end()) {
                            faceSampled.insert(facePositive);
                            faceSampled.insert(faceNegative);

                            int numOfVert = (int)vertices.size();
                            std::vector<int> face;
                            int co = 0;
                            for (int f = 0; f < points[fragment[k]].verticesFace[h].size(); f++)
                            {
                                int vertIndex = points[fragment[k]].verticesFace[h][f];
                                Eigen::Vector3d vertPos = points[fragment[k]].verticsCoor[vertIndex];

                                int vertIndexVertices = -999;
                                for (int fg = 0; fg < vertices.size(); fg++)
                                {
                                    Eigen::Vector3d existVert = vertices[fg];
                                    if (vertPos[0] == existVert[0] && vertPos[1] == existVert[1] && vertPos[2] == existVert[2])
                                    {
                                        vertIndexVertices = fg;
                                        break;
                                    }
                                }

                                if (newAlgoRemoveDup)
                                {
                                    int vertGlobalIndex = points[fragment[k]].globalIndexed[vertIndex];
                                    face.push_back(vertGlobalIndex);
                                }
                                else
                                {
                                    if (vertIndexVertices == -999)
                                    {
                                        vertices.push_back(vertPos);
                                        face.push_back(numOfVert + co);
                                        co += 1;
                                    }
                                    else {
                                        face.push_back(vertIndexVertices);
                                    }
                                }


                            }
                            faces.push_back(face);
                        }
                    }
                }
            }
        }

        bool findCrackSurface = true;
        meshObjFormat crackSurfacePartialCut;
        if (newAlgoRemoveDup)
        {
            // remove unnecessary points
            std::set<int> usedVerts;
            for (int face = 0; face < facesTmp.size(); ++face)
            {
                for (int vert = 0; vert < facesTmp[face].size(); ++vert)
                {
                    usedVerts.insert(facesTmp[face][vert]);
                }
            }
            std::vector<Eigen::Vector3d> usedVertsNewIndices;
            std::map<int, int> oldNewIndices; //key: old index; value: new index
            int numOfUsedVerts = 0;
            for (int vi = 0; vi < globalPoints.size(); vi++)
            {
                if (usedVerts.find(vi) != usedVerts.end()) // exists
                {
                    usedVertsNewIndices.push_back(globalPoints[vi]);
                    oldNewIndices[vi] = numOfUsedVerts;
                    numOfUsedVerts++;
                }
            }

            std::vector<std::vector<int>> reIndexedFaces;
            for (int face = 0; face < facesTmp.size(); ++face)
            {
                std::vector<int> ft;
                for (int vert = 0; vert < facesTmp[face].size(); ++vert)
                {
                    ft.push_back(oldNewIndices[facesTmp[face][vert]]);
                }
                reIndexedFaces.push_back(ft);
            }

            crackSurfacePartialCut.vertices = usedVertsNewIndices;
            crackSurfacePartialCut.faces = reIndexedFaces;


        }
        else
        {
            crackSurfacePartialCut.vertices = globalPoints;
            crackSurfacePartialCut.faces = facesTmp;
        }


        meshObjFormat crackSurfaceFullCut;
        if (newAlgoRemoveDup)
        {
            // remove unnecessary points
            std::set<int> usedVerts;
            for (int face = 0; face < faces.size(); ++face)
            {
                for (int vert = 0; vert < faces[face].size(); ++vert)
                {
                    usedVerts.insert(faces[face][vert]);
                }
            }
            std::vector<Eigen::Vector3d> usedVertsNewIndices;
            std::map<int, int> oldNewIndices; //key: old index; value: new index
            int numOfUsedVerts = 0;
            for (int vi = 0; vi < globalPoints.size(); vi++)
            {
                if (usedVerts.find(vi) != usedVerts.end()) // exists
                {
                    usedVertsNewIndices.push_back(globalPoints[vi]);
                    oldNewIndices[vi] = numOfUsedVerts;
                    numOfUsedVerts++;
                }
            }

            std::vector<std::vector<int>> reIndexedFaces;
            for (int face = 0; face < faces.size(); ++face)
            {
                std::vector<int> ft;
                for (int vert = 0; vert < faces[face].size(); ++vert)
                {
                    ft.push_back(oldNewIndices[faces[face][vert]]);
                }
                reIndexedFaces.push_back(ft);
            }

            crackSurfaceFullCut.vertices = usedVertsNewIndices;
            crackSurfaceFullCut.faces = reIndexedFaces;


        }
        else
        {
            crackSurfaceFullCut.vertices = vertices;
            crackSurfaceFullCut.faces = faces;
        }


        std::cout << "The number of crack faces is " << facesTmp.size() << std::endl;

        std::tuple<bool, meshObjFormat, meshObjFormat, std::vector<meshObjFormat>> resultReturn(findCrackSurface, crackSurfacePartialCut, crackSurfaceFullCut, allFragmentsObj);
        return resultReturn;
    }



}

#endif
