#include "ADTypes.hpp"

#include "even_polynomial.h"

namespace PHiLiP {
template <int dim, int nstate, typename real>
inline real Even_Polynomial<dim,nstate,real>
::value (const dealii::Point<dim,real> &point, const int istate) const
{
	real value = 0.0;
    for (int d=0; d<dim; d++) {
        value += pow(point[d] + 0.5, poly_max);
    }
    value += base_values[istate];
    return value;
}

template <int dim, int nstate, typename real>
inline dealii::Tensor<1,dim,real> Even_Polynomial<dim,nstate,real>
::gradient (const dealii::Point<dim,real> &point, const int /*istate*/) const
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

template <int dim, int nstate, typename real>
inline dealii::SymmetricTensor<2,dim,real> Even_Polynomial<dim,nstate,real>
::hessian (const dealii::Point<dim,real> &point, const int /*istate*/) const
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

template class Even_Polynomial<PHILIP_DIM, 1, double>;
template class Even_Polynomial<PHILIP_DIM, 2, double>;
template class Even_Polynomial<PHILIP_DIM, 3, double>;
template class Even_Polynomial<PHILIP_DIM, 4, double>;
template class Even_Polynomial<PHILIP_DIM, 5, double>;
template class Even_Polynomial<PHILIP_DIM, 8, double>;

template class Even_Polynomial<PHILIP_DIM, 1, FadType>;
template class Even_Polynomial<PHILIP_DIM, 2, FadType>;
template class Even_Polynomial<PHILIP_DIM, 3, FadType>;
template class Even_Polynomial<PHILIP_DIM, 4, FadType>;
template class Even_Polynomial<PHILIP_DIM, 5, FadType>;
template class Even_Polynomial<PHILIP_DIM, 8, FadType>;

template class Even_Polynomial<PHILIP_DIM, 1, RadType>;
template class Even_Polynomial<PHILIP_DIM, 2, RadType>;
template class Even_Polynomial<PHILIP_DIM, 3, RadType>;
template class Even_Polynomial<PHILIP_DIM, 4, RadType>;
template class Even_Polynomial<PHILIP_DIM, 5, RadType>;
template class Even_Polynomial<PHILIP_DIM, 8, RadType>;

template class Even_Polynomial<PHILIP_DIM, 1, FadFadType>;
template class Even_Polynomial<PHILIP_DIM, 2, FadFadType>;
template class Even_Polynomial<PHILIP_DIM, 3, FadFadType>;
template class Even_Polynomial<PHILIP_DIM, 4, FadFadType>;
template class Even_Polynomial<PHILIP_DIM, 5, FadFadType>;
template class Even_Polynomial<PHILIP_DIM, 8, FadFadType>;

template class Even_Polynomial<PHILIP_DIM, 1, RadFadType>;
template class Even_Polynomial<PHILIP_DIM, 2, RadFadType>;
template class Even_Polynomial<PHILIP_DIM, 3, RadFadType>;
template class Even_Polynomial<PHILIP_DIM, 4, RadFadType>;
template class Even_Polynomial<PHILIP_DIM, 5, RadFadType>;
template class Even_Polynomial<PHILIP_DIM, 8, RadFadType>;
} // PHiLiP namespace