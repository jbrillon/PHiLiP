#include "ADTypes.hpp"

#include "even_polynomial.h"

namespace PHiLiP {
template <int dim, typename real>
inline real Even_Polynomial<dim,real>
::value (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
	real value = 0.0;
    for (int d=0; d<dim; d++) {
        value += pow(point[d] + 0.5, poly_max);
    }
    value += base_values[istate];
    return value;
}

template <int dim, typename real>
inline dealii::Tensor<1,dim,real> Even_Polynomial<dim,real>
::gradient (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
	dealii::Tensor<1,dim,real> gradient;
	
    if (dim==1) {
        gradient[0] = poly_max*pow(point[0] + 0.5, poly_max-1);
    }
    if (dim==2) {
        gradient[0] = poly_max*pow(point[0] + 0.5, poly_max-1);
        gradient[1] = poly_max*pow(point[1] + 0.5, poly_max-1);
    }
    if (dim==3) {
        gradient[0] = poly_max*pow(point[0] + 0.5, poly_max-1);
        gradient[1] = poly_max*pow(point[1] + 0.5, poly_max-1);
        gradient[2] = poly_max*pow(point[2] + 0.5, poly_max-1);
    }

	return gradient;
}

template <int dim, typename real>
inline dealii::SymmetricTensor<2,dim,real> Even_Polynomial<dim,real>
::hessian (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    dealii::SymmetricTensor<2,dim,real> hessian;

    if (dim==1) {
        hessian[0][0] = poly_max*poly_max*pow(point[0] + 0.5, poly_max-2);
    }
    if (dim==2) {
        hessian[0][0] = poly_max*poly_max*pow(point[0] + 0.5, poly_max-2);
        hessian[0][1] = 0.0;

        hessian[1][0] = 0.0;
        hessian[1][1] = poly_max*poly_max*pow(point[1] + 0.5, poly_max-2);
    }
    if (dim==3) {
        hessian[0][0] = poly_max*poly_max*pow(point[0] + 0.5, poly_max-2);
        hessian[0][1] = 0.0;
        hessian[0][2] = 0.0;
        
        hessian[1][0] = 0.0;
        hessian[1][1] = poly_max*poly_max*pow(point[1] + 0.5, poly_max-2);
        hessian[1][2] = 0.0;
        
        hessian[2][0] = 0.0;
        hessian[2][1] = 0.0;
        hessian[2][2] = poly_max*poly_max*pow(point[2] + 0.5, poly_max-2);
    }

    return hessian;
}

template class Even_Polynomial<PHILIP_DIM, double>;
template class Even_Polynomial<PHILIP_DIM, FadType >;
template class Even_Polynomial<PHILIP_DIM, RadType >;
template class Even_Polynomial<PHILIP_DIM, FadFadType >;
template class Even_Polynomial<PHILIP_DIM, RadFadType >;

} // PHiLiP namespace