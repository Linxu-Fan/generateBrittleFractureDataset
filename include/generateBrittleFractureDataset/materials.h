#ifndef MATERIALS_H

#define MATERIALS_H

#include <math.h>
#include <Eigen/Core>

// Struct of material
struct Material 
{
	double density = 20; // particle density
	double E = 3.2e3; // Young's modulus
	double nu = 0.35; //Poisson ratio
	
	double mu = E / (2.0 * (1.0 + nu)); // lame parameter mu / shear modulus  ::::only for isotropic material
	double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)); // lame parameter lambda  ::::only for isotropic material
	double K = 2.0 / 3.0 * mu + lambda; // bulk modulus  ::::only for isotropic material


	// local damage field parameters
	double thetaf = 1.0e8;
	double Gf = 3;
	double lch = sqrt(2) * 0.04;
	double HsBar = thetaf * thetaf / 2.0 / E / Gf;
	double Hs = HsBar * lch / (1.0 - HsBar * lch);
	double damageThreshold = 0.97;
	double sigmoidK = 5; // this parameter control the curevature of the sigmoid function. It is recommend that it is bigger than 5


	// return mapping stress threshold(only for bending stress)
	double bendingStressThreshold = 1.0E6;




	void updateDenpendecies()
	{
		mu = E / (2.0 * (1.0 + nu)); // lame parameter mu / shear modulus  ::::only for isotropic material
		lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)); // lame parameter lambda  ::::only for isotropic material
		K = 2.0 / 3.0 * mu + lambda; // bulk modulus  ::::only for isotropic material

		HsBar = thetaf * thetaf / 2.0 / E / Gf;
		Hs = HsBar * lch / (1.0 - HsBar * lch);
	}

};


#endif













