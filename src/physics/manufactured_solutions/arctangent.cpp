#include "ADTypes.hpp"

#include "arctangent.h"

namespace PHiLiP {
template <int dim, int nstate, typename real>
inline real Arctangent<dim,nstate,real>
::value (const dealii::Point<dim,real> &point, const int /*istate*/) const
{
	real value = 1.0;
    for (int d=0; d<dim; d++) {
        const real dimval = atan(S1*(point[d]-loc1)) + atan(S2*(point[d]-loc2));
        value *= dimval;
        assert(isfinite(value));
    }
    return value;
}

template <int dim, int nstate, typename real>
inline dealii::Tensor<1,dim,real> Arctangent<dim,nstate,real>
::gradient (const dealii::Point<dim,real> &point, const int /*istate*/) const
{
	dealii::Tensor<1,dim,real> gradient;
	
	if (dim==1) {
        gradient[0]  = S1 / (pow(S1*(point[0]-loc1), 2) + 1.0);
        gradient[0] += S2 / (pow(S2*(point[0]-loc2), 2) + 1.0);
    }
    if (dim==2) {
        const real x = point[0], y = point[1];
        const real xval = atan(S1*(x-loc1)) + atan(S2*(x-loc2));
        const real yval = atan(S1*(y-loc1)) + atan(S2*(y-loc2));

        gradient[0]  = S1 / (pow(S1*(x-loc1), 2) + 1.0) + S2 / (pow(S2*(x-loc2), 2) + 1.0);
        gradient[0] *= yval;

        gradient[1]  = S1 / (pow(S1*(y-loc1), 2) + 1.0) + S2 / (pow(S2*(y-loc2), 2) + 1.0);
        gradient[1] *= xval;
    }
    if (dim==3) {
        const real xval = atan(S1*(point[0]-loc1)) + atan(S2*(point[0]-loc2));
        const real yval = atan(S1*(point[1]-loc1)) + atan(S2*(point[1]-loc2));
        const real zval = atan(S1*(point[2]-loc1)) + atan(S2*(point[2]-loc2));
        gradient[0]  = S1 / (pow(S1*(point[0]-loc1), 2) + 1.0);
        gradient[0] += S2 / (pow(S2*(point[0]-loc2), 2) + 1.0);
        gradient[0] *= yval;
        gradient[0] *= zval;
        gradient[1]  = S1 / (pow(S1*(point[1]-loc1), 2) + 1.0);
        gradient[1] += S2 / (pow(S2*(point[1]-loc2), 2) + 1.0);
        gradient[1] *= xval;
        gradient[1] *= zval;
        gradient[2]  = S1 / (pow(S1*(point[2]-loc1), 2) + 1.0);
        gradient[2] += S2 / (pow(S2*(point[2]-loc2), 2) + 1.0);
        gradient[2] *= xval;
        gradient[2] *= yval;
    }
	
	return gradient;
}

template <int dim, int nstate, typename real>
inline dealii::SymmetricTensor<2,dim,real> Arctangent<dim,nstate,real>
::hessian (const dealii::Point<dim,real> &point, const int /*istate*/) const
{
    dealii::SymmetricTensor<2,dim,real> hessian;

    dealii::Tensor<1,dim,real> gradient;
    if (dim==1) {
        const real x = point[0];
        gradient[0] = S1 / (pow(S1*(point[0]-loc1), 2) + 1.0);
        gradient[0] += S2 / (pow(S2*(point[0]-loc2), 2) + 1.0);

        hessian[0][0] = -(2* pow(S1,3)*(-loc1 + x))/pow(1 + pow(S1*(-loc1 + x),2),2)
                        -(2* pow(S2,3)*(-loc2 + x))/pow(1 + pow(S2*(-loc2 + x),2),2);
    }
    if (dim==2) {
        const real x = point[0], y = point[1];
        const real xval = atan(S1*(x-loc1)) + atan(S2*(x-loc2));
        const real yval = atan(S1*(y-loc1)) + atan(S2*(y-loc2));

        gradient[0]  = S1 / (pow(S1*(x-loc1), 2) + 1.0) + S2 / (pow(S2*(x-loc2), 2) + 1.0);
        gradient[0] *= yval;

        gradient[1]  = S1 / (pow(S1*(y-loc1), 2) + 1.0) + S2 / (pow(S2*(y-loc2), 2) + 1.0);
        gradient[1] *= xval;

        hessian[0][0] = -(2* pow(S1,3)*(-loc1 + x))/pow(1 + pow(S1*(-loc1 + x),2),2)
                        -(2* pow(S2,3)*(-loc2 + x))/pow(1 + pow(S2*(-loc2 + x),2),2);
        hessian[0][0] *= yval;

        hessian[0][1]  = S1 / (pow(S1*(x-loc1), 2) + 1.0);
        hessian[0][1] += S2 / (pow(S2*(x-loc2), 2) + 1.0);
        real temp = S1 / (pow(S1*(y-loc1), 2) + 1.0);
        temp += S2 / (pow(S2*(y-loc2), 2) + 1.0);
        hessian[0][1] *= temp;
        

        hessian[0][1] = (S1 / (pow(S1*(x-loc1), 2) + 1.0) +  S2 / (pow(S2*(x-loc2), 2) + 1.0));
        temp          = (S1 / (pow(S1*(y-loc1), 2) + 1.0) +  S2 / (pow(S2*(y-loc2), 2) + 1.0));
        hessian[0][1] *= temp;

        hessian[1][0] = hessian[0][1];

        hessian[1][1] = -(2* pow(S1,3)*(-loc1 + y))/pow(1 + pow(S1*(-loc1 + y),2),2)
                        -(2* pow(S2,3)*(-loc2 + y))/pow(1 + pow(S2*(-loc2 + y),2),2);
        hessian[1][1] *= xval;

    }
    if (dim==3) {
        std::abort();
    }

    return hessian;
}

template class Arctangent<PHILIP_DIM, 1, double>;
template class Arctangent<PHILIP_DIM, 2, double>;
template class Arctangent<PHILIP_DIM, 3, double>;
template class Arctangent<PHILIP_DIM, 4, double>;
template class Arctangent<PHILIP_DIM, 5, double>;
template class Arctangent<PHILIP_DIM, 8, double>;

template class Arctangent<PHILIP_DIM, 1, FadType>;
template class Arctangent<PHILIP_DIM, 2, FadType>;
template class Arctangent<PHILIP_DIM, 3, FadType>;
template class Arctangent<PHILIP_DIM, 4, FadType>;
template class Arctangent<PHILIP_DIM, 5, FadType>;
template class Arctangent<PHILIP_DIM, 8, FadType>;

template class Arctangent<PHILIP_DIM, 1, RadType>;
template class Arctangent<PHILIP_DIM, 2, RadType>;
template class Arctangent<PHILIP_DIM, 3, RadType>;
template class Arctangent<PHILIP_DIM, 4, RadType>;
template class Arctangent<PHILIP_DIM, 5, RadType>;
template class Arctangent<PHILIP_DIM, 8, RadType>;

template class Arctangent<PHILIP_DIM, 1, FadFadType>;
template class Arctangent<PHILIP_DIM, 2, FadFadType>;
template class Arctangent<PHILIP_DIM, 3, FadFadType>;
template class Arctangent<PHILIP_DIM, 4, FadFadType>;
template class Arctangent<PHILIP_DIM, 5, FadFadType>;
template class Arctangent<PHILIP_DIM, 8, FadFadType>;

template class Arctangent<PHILIP_DIM, 1, RadFadType>;
template class Arctangent<PHILIP_DIM, 2, RadFadType>;
template class Arctangent<PHILIP_DIM, 3, RadFadType>;
template class Arctangent<PHILIP_DIM, 4, RadFadType>;
template class Arctangent<PHILIP_DIM, 5, RadFadType>;
template class Arctangent<PHILIP_DIM, 8, RadFadType>;
} // PHiLiP namespace