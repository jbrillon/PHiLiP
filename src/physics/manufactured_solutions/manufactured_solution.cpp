#include <CoDiPack/include/codi.hpp>
#include <Sacado.hpp>

#include <deal.II/base/function.h>
#include <deal.II/base/function.templates.h> // Needed to instantiate dealii::Function<PHILIP_DIM,Sacado::Fad::DFad<double>>
#include <deal.II/base/function_time.templates.h> // Needed to instantiate dealii::Function<PHILIP_DIM,Sacado::Fad::DFad<double>>

#include "manufactured_solution.h"

//template class dealii::FunctionTime<Sacado::Fad::DFad<double>>; // Needed by Function
//template class dealii::Function<PHILIP_DIM,Sacado::Fad::DFad<double>>;

namespace PHiLiP {

// ///< Provide isfinite for double.
// bool isfinite(double value)
// {
//     return std::isfinite(static_cast<double>(value));
// }

// ///< Provide isfinite for FadType
// bool isfinite(Sacado::Fad::DFad<double> value)
// {
//     return std::isfinite(static_cast<double>(value.val()));
// }

// ///< Provide isfinite for FadFadType
// bool isfinite(Sacado::Fad::DFad<Sacado::Fad::DFad<double>> value)
// {
//     return std::isfinite(static_cast<double>(value.val().val()));
// }

// ///< Provide isfinite for RadFadType
// bool isfinite(Sacado::Rad::ADvar<Sacado::Fad::DFad<double>> value)
// {
//     return std::isfinite(static_cast<double>(value.val().val()));
// }

template <int dim, int nstate, typename real>
ManufacturedSolutionFunction<dim,nstate,real>
::ManufacturedSolutionFunction()
    : dealii::Function<dim,real>(nstate)
    , base_values(nstate)
    , amplitudes(nstate)
    , frequencies(nstate)
{
    const double pi = atan(1)*4.0;

    for (int s=0; s<nstate; s++) {
        base_values[s] = 1+(s+1.0)/nstate;
        base_values[nstate-1] = 10;
        amplitudes[s] = 0.2*base_values[s]*sin((static_cast<double>(nstate)-s)/nstate);
        for (int d=0; d<dim; d++) {
            frequencies[s][d] = 2.0 + sin(0.1+s*0.5+d*0.2) *  pi / 2.0;
        }
    }
}

// template <int dim, int nstate, typename real>
// ManufacturedSolutionFunction<dim,nstate,real>::~ManufacturedSolutionFunction() {}

// template <int dim, int nstate, typename real>
// inline dealii::Tensor<1,dim,real> ManufacturedSolutionFunction<dim,nstate,real>
// ::gradient (const dealii::Point<dim,real> &point, const int istate) const
// {
    // dealii::Tensor<1,dim,real> gradient;
    // for (int dim_deri=0; dim_deri<dim; dim_deri++) {
    //     gradient[dim_deri] = amplitudes[istate] * frequencies[istate][dim_deri];
    //     for (int dim_trig=0; dim_trig<dim; dim_trig++) {
    //         const real angle = frequencies[istate][dim_trig] * point[dim_trig];
    //         if (dim_deri == dim_trig) gradient[dim_deri] *= cos( angle );
    //         if (dim_deri != dim_trig) gradient[dim_deri] *= sin( angle );
    //     }
    //     assert(isfinite(gradient[dim_deri]));
    // }
//     return gradient;
// }

template <int dim, int nstate, typename real>
inline dealii::Tensor<1,dim,real> ManufacturedSolutionFunction<dim,nstate,real>
::gradient_fd (const dealii::Point<dim,real> &point, const int istate) const
{
    dealii::Tensor<1,dim,real> gradient;
    const double eps=1e-6;
    for (int dim_deri=0; dim_deri<dim; dim_deri++) {
        dealii::Point<dim,real> pert_p = point;
        dealii::Point<dim,real> pert_m = point;
        pert_p[dim_deri] += eps;
        pert_m[dim_deri] -= eps;
        const real value_p = value(pert_p,istate);
        const real value_m = value(pert_m,istate);
        gradient[dim_deri] = (value_p - value_m) / (2*eps);
    }
    return gradient;
}

template <int dim, int nstate, typename real>
inline dealii::SymmetricTensor<2,dim,real> ManufacturedSolutionFunction<dim,nstate,real>
::hessian_fd (const dealii::Point<dim,real> &point, const int istate) const
{
    dealii::SymmetricTensor<2,dim,real> hessian;
    const double eps=1e-4;
    for (int d1=0; d1<dim; d1++) {
        for (int d2=d1; d2<dim; d2++) {
            dealii::Point<dim,real> pert_p_p = point;
            dealii::Point<dim,real> pert_p_m = point;
            dealii::Point<dim,real> pert_m_p = point;
            dealii::Point<dim,real> pert_m_m = point;

            pert_p_p[d1] += (+eps); pert_p_p[d2] += (+eps);
            pert_p_m[d1] += (+eps); pert_p_m[d2] += (-eps);
            pert_m_p[d1] += (-eps); pert_m_p[d2] += (+eps);
            pert_m_m[d1] += (-eps); pert_m_m[d2] += (-eps);

            const real valpp = value(pert_p_p, istate);
            const real valpm = value(pert_p_m, istate);
            const real valmp = value(pert_m_p, istate);
            const real valmm = value(pert_m_m, istate);

            hessian[d1][d2] = (valpp - valpm - valmp + valmm) / (4*eps*eps);
        }
    }
    return hessian;
}

template <int dim, int nstate, typename real>
void ManufacturedSolutionFunction<dim,nstate,real>
::vector_gradient (
    const dealii::Point<dim,real> &p,
    std::vector<dealii::Tensor<1,dim, real> > &gradients) const
{
    for (int i = 0; i < nstate; ++i)
        gradients[i] = gradient(p, i);
}


template <int dim, int nstate, typename real>
inline std::vector<real> ManufacturedSolutionFunction<dim,nstate,real>
::stdvector_values (const dealii::Point<dim,real> &point) const
{
    std::vector<real> values(nstate);
    for (int s=0; s<nstate; s++) { values[s] = value(point, s); }
    return values;
}

using FadType = Sacado::Fad::DFad<double>; ///< Sacado AD type for first derivatives.
using FadFadType = Sacado::Fad::DFad<FadType>; ///< Sacado AD type that allows 2nd derivatives.

static constexpr int dimForwardAD = 1; ///< Size of the forward vector mode for CoDiPack.
static constexpr int dimReverseAD = 1; ///< Size of the reverse vector mode for CoDiPack.

using codi_FadType = codi::RealForwardGen<double, codi::Direction<double,dimForwardAD>>; ///< Tapeless forward mode.
//using codi_FadType = codi::RealForwardGen<double, codi::DirectionVar<double>>;

using codi_JacobianComputationType = codi::RealReverseIndexVec<dimReverseAD>; ///< Reverse mode type for Jacobian computation using TapeHelper.
using codi_HessianComputationType = codi::RealReversePrimalIndexGen< codi::RealForwardVec<dimForwardAD>,
                                                  codi::Direction< codi::RealForwardVec<dimForwardAD>, dimReverseAD>
                                                >; ///< Nested reverse-forward mode type for Jacobian and Hessian computation using TapeHelper.
//using RadFadType = Sacado::Rad::ADvar<FadType>; ///< Sacado AD type that allows 2nd derivatives.
//using RadFadType = codi_JacobianComputationType; ///< Reverse only mode that only allows Jacobian computation.
using RadType = codi_JacobianComputationType; ///< CoDiPaco reverse-AD type for first derivatives.
using RadFadType = codi_HessianComputationType; ///< Nested reverse-forward mode type for Jacobian and Hessian computation using TapeHelper.

template class ManufacturedSolutionFunction<PHILIP_DIM, 1, double>;
template class ManufacturedSolutionFunction<PHILIP_DIM, 2, double>;
template class ManufacturedSolutionFunction<PHILIP_DIM, 3, double>;
template class ManufacturedSolutionFunction<PHILIP_DIM, 4, double>;
template class ManufacturedSolutionFunction<PHILIP_DIM, 5, double>;
template class ManufacturedSolutionFunction<PHILIP_DIM, 8, double>;

template class ManufacturedSolutionFunction<PHILIP_DIM, 1, FadType>;
template class ManufacturedSolutionFunction<PHILIP_DIM, 2, FadType>;
template class ManufacturedSolutionFunction<PHILIP_DIM, 3, FadType>;
template class ManufacturedSolutionFunction<PHILIP_DIM, 4, FadType>;
template class ManufacturedSolutionFunction<PHILIP_DIM, 5, FadType>;
template class ManufacturedSolutionFunction<PHILIP_DIM, 8, FadType>;

template class ManufacturedSolutionFunction<PHILIP_DIM, 1, RadType>;
template class ManufacturedSolutionFunction<PHILIP_DIM, 2, RadType>;
template class ManufacturedSolutionFunction<PHILIP_DIM, 3, RadType>;
template class ManufacturedSolutionFunction<PHILIP_DIM, 4, RadType>;
template class ManufacturedSolutionFunction<PHILIP_DIM, 5, RadType>;
template class ManufacturedSolutionFunction<PHILIP_DIM, 8, RadType>;

template class ManufacturedSolutionFunction<PHILIP_DIM, 1, FadFadType>;
template class ManufacturedSolutionFunction<PHILIP_DIM, 2, FadFadType>;
template class ManufacturedSolutionFunction<PHILIP_DIM, 3, FadFadType>;
template class ManufacturedSolutionFunction<PHILIP_DIM, 4, FadFadType>;
template class ManufacturedSolutionFunction<PHILIP_DIM, 5, FadFadType>;
template class ManufacturedSolutionFunction<PHILIP_DIM, 8, FadFadType>;

template class ManufacturedSolutionFunction<PHILIP_DIM, 1, RadFadType>;
template class ManufacturedSolutionFunction<PHILIP_DIM, 2, RadFadType>;
template class ManufacturedSolutionFunction<PHILIP_DIM, 3, RadFadType>;
template class ManufacturedSolutionFunction<PHILIP_DIM, 4, RadFadType>;
template class ManufacturedSolutionFunction<PHILIP_DIM, 5, RadFadType>;
template class ManufacturedSolutionFunction<PHILIP_DIM, 8, RadFadType>;
}