#include "ADTypes.hpp"

#include "exponential.h"

namespace PHiLiP {
template <int dim, int nstate, typename real>
inline real Exponential<dim,nstate,real>
::value (const dealii::Point<dim,real> &point, const int istate) const
{
	real value = 0.0;
    for (int d=0; d<dim; d++) {
        value += exp( point[d] );
        assert(isfinite(value));
    }
    value += base_values[istate];
    return value;
}

template <int dim, int nstate, typename real>
inline dealii::Tensor<1,dim,real> Exponential<dim,nstate,real>
::gradient (const dealii::Point<dim,real> &point, const int /*istate*/) const
{
	dealii::Tensor<1,dim,real> gradient;
	
    if (dim==1) {
	    gradient[0] = exp(point[0]);
	}
	if (dim==2) {
	    gradient[0] = exp(point[0]);
	    gradient[1] = exp(point[1]);
	}
	if (dim==3) {
	    gradient[0] = exp(point[0]);
	    gradient[1] = exp(point[1]);
	    gradient[2] = exp(point[2]);
	}

	return gradient;	
}

template <int dim, int nstate, typename real>
inline dealii::SymmetricTensor<2,dim,real> Exponential<dim,nstate,real>
::hessian (const dealii::Point<dim,real> &point, const int /*istate*/) const
{
    dealii::SymmetricTensor<2,dim,real> hessian;

    if (dim==1) {
        hessian[0][0] = exp(point[0]);
    }
    if (dim==2) {
        hessian[0][0] = exp(point[0]);
        hessian[0][1] = 0.0;

        hessian[1][0] = 0.0;
        hessian[1][1] = exp(point[1]);
    }
    if (dim==3) {
        hessian[0][0] = exp(point[0]);
        hessian[0][1] = 0.0;
        hessian[0][2] = 0.0;
        
        hessian[1][0] = 0.0;
        hessian[1][1] = exp(point[1]);
        hessian[1][2] = 0.0;
        
        hessian[2][0] = 0.0;
        hessian[2][1] = 0.0;
        hessian[2][2] = exp(point[2]);
    }

    return hessian;
}

template class Exponential<PHILIP_DIM, 1, double>;
template class Exponential<PHILIP_DIM, 2, double>;
template class Exponential<PHILIP_DIM, 3, double>;
template class Exponential<PHILIP_DIM, 4, double>;
template class Exponential<PHILIP_DIM, 5, double>;
template class Exponential<PHILIP_DIM, 8, double>;

template class Exponential<PHILIP_DIM, 1, FadType>;
template class Exponential<PHILIP_DIM, 2, FadType>;
template class Exponential<PHILIP_DIM, 3, FadType>;
template class Exponential<PHILIP_DIM, 4, FadType>;
template class Exponential<PHILIP_DIM, 5, FadType>;
template class Exponential<PHILIP_DIM, 8, FadType>;

template class Exponential<PHILIP_DIM, 1, RadType>;
template class Exponential<PHILIP_DIM, 2, RadType>;
template class Exponential<PHILIP_DIM, 3, RadType>;
template class Exponential<PHILIP_DIM, 4, RadType>;
template class Exponential<PHILIP_DIM, 5, RadType>;
template class Exponential<PHILIP_DIM, 8, RadType>;

template class Exponential<PHILIP_DIM, 1, FadFadType>;
template class Exponential<PHILIP_DIM, 2, FadFadType>;
template class Exponential<PHILIP_DIM, 3, FadFadType>;
template class Exponential<PHILIP_DIM, 4, FadFadType>;
template class Exponential<PHILIP_DIM, 5, FadFadType>;
template class Exponential<PHILIP_DIM, 8, FadFadType>;

template class Exponential<PHILIP_DIM, 1, RadFadType>;
template class Exponential<PHILIP_DIM, 2, RadFadType>;
template class Exponential<PHILIP_DIM, 3, RadFadType>;
template class Exponential<PHILIP_DIM, 4, RadFadType>;
template class Exponential<PHILIP_DIM, 5, RadFadType>;
template class Exponential<PHILIP_DIM, 8, RadFadType>;

} // PHiLiP namespace