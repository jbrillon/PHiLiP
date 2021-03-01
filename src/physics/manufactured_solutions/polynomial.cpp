#include "ADTypes.hpp"

#include "polynomial.h"

namespace PHiLiP {
template <int dim, typename real>
inline real Polynomial<dim,real>
::value (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
	real value = 0.0;
    for (int d=0; d<dim; d++) {
        const real x = point[d];
        value += 1.0 + x - x*x - x*x*x + x*x*x*x - x*x*x*x*x + x*x*x*x*x*x + 0.001*sin(50*x);
    }
    value += base_values[istate];
    return value;
}

template <int dim, typename real>
inline dealii::Tensor<1,dim,real> Polynomial<dim,real>
::gradient (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
	dealii::Tensor<1,dim,real> gradient;
	
	if (dim==1) {
        const real x = point[0];
        gradient[0] = 1.0 - 2*x -3*x*x + 4*x*x*x - 5*x*x*x*x + 6*x*x*x*x*x + 0.050*cos(50*x);
    }
    if (dim==2) {
        real x = point[0];
        gradient[0] = 1.0 - 2*x -3*x*x + 4*x*x*x - 5*x*x*x*x + 6*x*x*x*x*x + 0.050*cos(50*x);
        x = point[1];
        gradient[1] = 1.0 - 2*x -3*x*x + 4*x*x*x - 5*x*x*x*x + 6*x*x*x*x*x + 0.050*cos(50*x);
    }
    if (dim==3) {
        real x = point[0];
        gradient[0] = 1.0 - 2*x -3*x*x + 4*x*x*x - 5*x*x*x*x + 6*x*x*x*x*x;
        x = point[1];
        gradient[1] = 1.0 - 2*x -3*x*x + 4*x*x*x - 5*x*x*x*x + 6*x*x*x*x*x;
        x = point[2];
        gradient[2] = 1.0 - 2*x -3*x*x + 4*x*x*x - 5*x*x*x*x + 6*x*x*x*x*x;
    }

	return gradient;
}

template <int dim, typename real>
inline dealii::SymmetricTensor<2,dim,real> Polynomial<dim,real>
::hessian (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    dealii::SymmetricTensor<2,dim,real> hessian;

    if (dim==1) {
        const real x = point[0];
        hessian[0][0] = - 2.0 -6*x + 12*x*x - 20*x*x*x + 30*x*x*x*x - 2.500*sin(50*x);
    }
    if (dim==2) {
        real x = point[0];
        hessian[0][0] = - 2.0 -6*x + 12*x*x - 20*x*x*x + 30*x*x*x*x - 2.500*sin(50*x);
        x = point[1];
        hessian[1][1] = - 2.0 -6*x + 12*x*x - 20*x*x*x + 30*x*x*x*x - 2.500*sin(50*x);
    }
    if (dim==3) {
        real x = point[0];
        hessian[0][0] = - 2.0 -6*x + 12*x*x - 20*x*x*x + 30*x*x*x*x - 2.500*sin(50*x);
        x = point[1];
        hessian[1][1] = - 2.0 -6*x + 12*x*x - 20*x*x*x + 30*x*x*x*x - 2.500*sin(50*x);
        x = point[2];
        hessian[2][2] = - 2.0 -6*x + 12*x*x - 20*x*x*x + 30*x*x*x*x - 2.500*sin(50*x);
    }

    return hessian;
}

template class Polynomial<PHILIP_DIM, double>;
template class Polynomial<PHILIP_DIM, FadType >;
template class Polynomial<PHILIP_DIM, RadType >;
template class Polynomial<PHILIP_DIM, FadFadType >;
template class Polynomial<PHILIP_DIM, RadFadType >;

} // PHiLiP namespace