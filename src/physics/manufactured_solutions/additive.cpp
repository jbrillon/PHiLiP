#include "ADTypes.hpp"

#include "additive.h"

namespace PHiLiP {
template <int dim, int nstate, typename real>
inline real Additive<dim,nstate,real>
::value (const dealii::Point<dim,real> &point, const int istate) const
{
	real value = 0.0;
    for (int d=0; d<dim; d++) {
        value += amplitudes[istate]*sin( frequencies[istate][d] * point[d] );
        assert(isfinite(value));
    }
    value += base_values[istate];
    return value;
}

template <int dim, int nstate, typename real>
inline dealii::Tensor<1,dim,real> Additive<dim,nstate,real>
::gradient (const dealii::Point<dim,real> &point, const int istate) const
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

template <int dim, int nstate, typename real>
inline dealii::SymmetricTensor<2,dim,real> Additive<dim,nstate,real>
::hessian (const dealii::Point<dim,real> &point, const int istate) const
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

template class Additive<PHILIP_DIM, 1, double>;
template class Additive<PHILIP_DIM, 2, double>;
template class Additive<PHILIP_DIM, 3, double>;
template class Additive<PHILIP_DIM, 4, double>;
template class Additive<PHILIP_DIM, 5, double>;
template class Additive<PHILIP_DIM, 8, double>;

template class Additive<PHILIP_DIM, 1, FadType>;
template class Additive<PHILIP_DIM, 2, FadType>;
template class Additive<PHILIP_DIM, 3, FadType>;
template class Additive<PHILIP_DIM, 4, FadType>;
template class Additive<PHILIP_DIM, 5, FadType>;
template class Additive<PHILIP_DIM, 8, FadType>;

template class Additive<PHILIP_DIM, 1, RadType>;
template class Additive<PHILIP_DIM, 2, RadType>;
template class Additive<PHILIP_DIM, 3, RadType>;
template class Additive<PHILIP_DIM, 4, RadType>;
template class Additive<PHILIP_DIM, 5, RadType>;
template class Additive<PHILIP_DIM, 8, RadType>;

template class Additive<PHILIP_DIM, 1, FadFadType>;
template class Additive<PHILIP_DIM, 2, FadFadType>;
template class Additive<PHILIP_DIM, 3, FadFadType>;
template class Additive<PHILIP_DIM, 4, FadFadType>;
template class Additive<PHILIP_DIM, 5, FadFadType>;
template class Additive<PHILIP_DIM, 8, FadFadType>;

template class Additive<PHILIP_DIM, 1, RadFadType>;
template class Additive<PHILIP_DIM, 2, RadFadType>;
template class Additive<PHILIP_DIM, 3, RadFadType>;
template class Additive<PHILIP_DIM, 4, RadFadType>;
template class Additive<PHILIP_DIM, 5, RadFadType>;
template class Additive<PHILIP_DIM, 8, RadFadType>;

} // PHiLiP namespace