#include "ADTypes.hpp"

#include "additive.h"

namespace PHiLiP {
template <int dim, typename real>
inline real Additive<dim,real>
::value (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
	real value = 0.0;
    for (int d=0; d<dim; d++) {
        value += amplitudes[istate]*sin( frequencies[istate][d] * point[d] );
        assert(isfinite(value));
    }
    value += base_values[istate];
    return value;
}

template <int dim, typename real>
inline dealii::Tensor<1,dim,real> Additive<dim,real>
::gradient (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
	dealii::Tensor<1,dim,real> gradient;
	
	const real A = amplitudes[istate];
    const dealii::Tensor<1,dim,real> f = frequencies[istate];

    if (dim==1) {
        const real fx = f[0]*point[0];
        gradient[0] = A*f[0]*cos(fx);
    }
    if (dim==2) {
        const real fx = f[0]*point[0];
        const real fy = f[1]*point[1];
        gradient[0] = A*f[0]*cos(fx);
        gradient[1] = A*f[1]*cos(fy);
    }
    if (dim==3) {
        const real fx = f[0]*point[0];
        const real fy = f[1]*point[1];
        const real fz = f[2]*point[2];
        gradient[0] = A*f[0]*cos(fx);
        gradient[1] = A*f[1]*cos(fy);
        gradient[2] = A*f[2]*cos(fz);
    }

	return gradient;	
}

template <int dim, typename real>
inline dealii::SymmetricTensor<2,dim,real> Additive<dim,real>
::hessian (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    dealii::SymmetricTensor<2,dim,real> hessian;

    const real A = amplitudes[istate];
    const dealii::Tensor<1,dim,real> f = frequencies[istate];

    if (dim==1) {
        const real fx = f[0]*point[0];
        hessian[0][0] = -A*f[0]*f[0]*sin(fx);
    }
    if (dim==2) {
        const real fx = f[0]*point[0];
        const real fy = f[1]*point[1];
        hessian[0][0] = -A*f[0]*f[0]*sin(fx);
        hessian[0][1] =  0.0;

        hessian[1][0] =  0.0;
        hessian[1][1] = -A*f[1]*f[1]*sin(fy);
    }
    if (dim==3) {
        const real fx = f[0]*point[0];
        const real fy = f[1]*point[1];
        const real fz = f[2]*point[2];
        hessian[0][0] = -A*f[0]*f[0]*sin(fx);
        hessian[0][1] =  0.0;
        hessian[0][2] =  0.0;
        
        hessian[1][0] =  0.0;
        hessian[1][1] = -A*f[1]*f[1]*sin(fy);
        hessian[1][2] =  0.0;
        
        hessian[2][0] =  0.0;
        hessian[2][1] =  0.0;
        hessian[2][2] = -A*f[2]*f[2]*sin(fz);
    }

    return hessian;
}


template class Additive<PHILIP_DIM, double>;
template class Additive<PHILIP_DIM, FadType >;
template class Additive<PHILIP_DIM, RadType >;
template class Additive<PHILIP_DIM, FadFadType >;
template class Additive<PHILIP_DIM, RadFadType >;

} // PHiLiP namespace