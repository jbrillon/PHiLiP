#include "ADTypes.hpp"

#include "sine.h"

namespace PHiLiP {
template <int dim, typename real>
inline real Sine<dim,real>
::value (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
	real value = amplitudes[istate];
    for (int d=0; d<dim; d++) {
        value *= sin( frequencies[istate][d] * point[d] );
        assert(isfinite(value));
    }
    value += base_values[istate];
    return value;
}

template <int dim, typename real>
inline dealii::Tensor<1,dim,real> Sine<dim,real>
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
        gradient[0] = A*f[0]*cos(fx)*sin(fy);
        gradient[1] = A*f[1]*sin(fx)*cos(fy);
    }
    if (dim==3) {
        const real fx = f[0]*point[0];
        const real fy = f[1]*point[1];
        const real fz = f[2]*point[2];
        gradient[0] = A*f[0]*cos(fx)*sin(fy)*sin(fz);
        gradient[1] = A*f[1]*sin(fx)*cos(fy)*sin(fz);
        gradient[2] = A*f[2]*sin(fx)*sin(fy)*cos(fz);
    }
    
    return gradient;
}

template <int dim, typename real>
inline dealii::SymmetricTensor<2,dim,real> Sine<dim,real>
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
        hessian[0][0] = -A*f[0]*f[0]*sin(fx)*sin(fy);
        hessian[0][1] =  A*f[0]*f[1]*cos(fx)*cos(fy);

        hessian[1][0] =  A*f[1]*f[0]*cos(fx)*cos(fy);
        hessian[1][1] = -A*f[1]*f[1]*sin(fx)*sin(fy);
    }
    if (dim==3) {
        const real fx = f[0]*point[0];
        const real fy = f[1]*point[1];
        const real fz = f[2]*point[2];
        hessian[0][0] = -A*f[0]*f[0]*sin(fx)*sin(fy)*sin(fz);
        hessian[0][1] =  A*f[0]*f[1]*cos(fx)*cos(fy)*sin(fz);
        hessian[0][2] =  A*f[0]*f[2]*cos(fx)*sin(fy)*cos(fz);
        
        hessian[1][0] =  A*f[1]*f[0]*cos(fx)*cos(fy)*sin(fz);
        hessian[1][1] = -A*f[1]*f[1]*sin(fx)*sin(fy)*sin(fz);
        hessian[1][2] =  A*f[1]*f[2]*sin(fx)*cos(fy)*cos(fz);
        
        hessian[2][0] =  A*f[2]*f[0]*cos(fx)*sin(fy)*cos(fz);
        hessian[2][1] =  A*f[2]*f[1]*sin(fx)*cos(fy)*cos(fz);
        hessian[2][2] = -A*f[2]*f[2]*sin(fx)*sin(fy)*sin(fz);
    }

    return hessian;
}

template class Sine<PHILIP_DIM, double>;
template class Sine<PHILIP_DIM, FadType >;
template class Sine<PHILIP_DIM, RadType >;
template class Sine<PHILIP_DIM, FadFadType >;
template class Sine<PHILIP_DIM, RadFadType >;

} // PHiLiP namespace