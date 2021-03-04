#include "ADTypes.hpp"

#include "polynomial.h"

namespace PHiLiP {
template <int dim, int nstate, typename real>
inline real Polynomial<dim,nstate,real>
::value (const dealii::Point<dim,real> &point, const int istate) const
{
	real value = 0.0;
    for (int d=0; d<dim; d++) {
        const real x = point[d];
        value += 1.0 + x - x*x - x*x*x + x*x*x*x - x*x*x*x*x + x*x*x*x*x*x + 0.001*sin(50*x);
    }
    value += base_values[istate];
    return value;
}

template <int dim, int nstate, typename real>
inline dealii::Tensor<1,dim,real> Polynomial<dim,nstate,real>
::gradient (const dealii::Point<dim,real> &point, const int /*istate*/) const
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

template <int dim, int nstate, typename real>
inline dealii::SymmetricTensor<2,dim,real> Polynomial<dim,nstate,real>
::hessian (const dealii::Point<dim,real> &point, const int /*istate*/) const
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

template class Polynomial<PHILIP_DIM, 1, double>;
template class Polynomial<PHILIP_DIM, 2, double>;
template class Polynomial<PHILIP_DIM, 3, double>;
template class Polynomial<PHILIP_DIM, 4, double>;
template class Polynomial<PHILIP_DIM, 5, double>;
template class Polynomial<PHILIP_DIM, 8, double>;

template class Polynomial<PHILIP_DIM, 1, FadType>;
template class Polynomial<PHILIP_DIM, 2, FadType>;
template class Polynomial<PHILIP_DIM, 3, FadType>;
template class Polynomial<PHILIP_DIM, 4, FadType>;
template class Polynomial<PHILIP_DIM, 5, FadType>;
template class Polynomial<PHILIP_DIM, 8, FadType>;

template class Polynomial<PHILIP_DIM, 1, RadType>;
template class Polynomial<PHILIP_DIM, 2, RadType>;
template class Polynomial<PHILIP_DIM, 3, RadType>;
template class Polynomial<PHILIP_DIM, 4, RadType>;
template class Polynomial<PHILIP_DIM, 5, RadType>;
template class Polynomial<PHILIP_DIM, 8, RadType>;

template class Polynomial<PHILIP_DIM, 1, FadFadType>;
template class Polynomial<PHILIP_DIM, 2, FadFadType>;
template class Polynomial<PHILIP_DIM, 3, FadFadType>;
template class Polynomial<PHILIP_DIM, 4, FadFadType>;
template class Polynomial<PHILIP_DIM, 5, FadFadType>;
template class Polynomial<PHILIP_DIM, 8, FadFadType>;

template class Polynomial<PHILIP_DIM, 1, RadFadType>;
template class Polynomial<PHILIP_DIM, 2, RadFadType>;
template class Polynomial<PHILIP_DIM, 3, RadFadType>;
template class Polynomial<PHILIP_DIM, 4, RadFadType>;
template class Polynomial<PHILIP_DIM, 5, RadFadType>;
template class Polynomial<PHILIP_DIM, 8, RadFadType>;

} // PHiLiP namespace