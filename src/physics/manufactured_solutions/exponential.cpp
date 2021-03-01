#include "ADTypes.hpp"

#include "exponential.h"

namespace PHiLiP {
template <int dim, typename real>
inline real Exponential<dim,real>
::value (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
	real value = 0.0;
    for (int d=0; d<dim; d++) {
        value += exp( point[d] );
        assert(isfinite(value));
    }
    value += base_values[istate];
    return value;
}

template <int dim, typename real>
inline dealii::Tensor<1,dim,real> Exponential<dim,real>
::gradient (const dealii::Point<dim,real> &point, const unsigned int istate) const
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

template <int dim, typename real>
inline dealii::SymmetricTensor<2,dim,real> Exponential<dim,real>
::hessian (const dealii::Point<dim,real> &point, const unsigned int istate) const
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

template class Exponential<PHILIP_DIM, double>;
template class Exponential<PHILIP_DIM, FadType >;
template class Exponential<PHILIP_DIM, RadType >;
template class Exponential<PHILIP_DIM, FadFadType >;
template class Exponential<PHILIP_DIM, RadFadType >;

} // PHiLiP namespace