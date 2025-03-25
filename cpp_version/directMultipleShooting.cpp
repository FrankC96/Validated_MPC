#include <iostream>
#include "ariadne.hpp"
#include "utils.hpp"
#include "ariadne/solvers/runge_kutta_integrator.hpp"
#include "ariadne/solvers/nonlinear_programming.hpp"

using namespace Ariadne;
using VectorAndScalarPatch = Pair<ValidatedVectorMultivariateFunctionPatch, ValidatedScalarMultivariateFunctionPatch>;

FloatDPBoundsVector optimization_step(EffectiveVectorMultivariateFunction f, BoxDomainType x0dom, BoxDomainType xdom, IntervalDomainType udom, int num_shooting_nodes, StepSizeType h)
{
    RealVariable t("t");
    IntervalDomainType tdom = { 0, h };

    IntervalDomainType cdom = { -0.0_x, +0.0_x };

    List<RealVectorVariable> X = {};
    List<RealVariable> U;

    List<BoxDomainType> x_doms = {};
    List<IntervalDomainType> u_doms = {};
    List<IntervalDomainType> c_doms = {};

    List<VariableIntervalOrBoxDomainType> xu_restr_domains;

    List<ValidatedVectorMultivariateFunctionPatch> phi_patches;
    List<ValidatedScalarMultivariateFunctionPatch> gamma_patches;

    List<ValidatedScalarMultivariateFunctionPatch> constraints;

    ValidatedVectorMultivariateFunctionPatch step;
    ValidatedVectorMultivariateFunctionPatch state;
    ValidatedVectorMultivariateFunctionPatch g;
    ValidatedScalarMultivariateFunctionPatch objective;

    for (int i = 0;i <= num_shooting_nodes;++i)
    {
        RealVectorVariable x = RealVectorVariable("x" + to_string(i), 2);
        RealVariable u = RealVariable("u" + to_string(i));
        RealVariable t = RealVariable("t");

        X.append(x);
        U.append(u);
    }

    x_doms.append(x0dom);
    u_doms.append(udom);
    for (int i = 0;i <= num_shooting_nodes;++i)
    {
        x_doms.append(xdom);
        u_doms.append(udom);

        VectorVariableBoxDomainType xr = X[i] | x_doms[i];
        VariableIntervalDomainType ur = U[i] | u_doms[i];
        VariableIntervalDomainType tr = t | tdom;

        ValidatedVectorMultivariateFunctionPatch phigamma_ = integrateDynamics(f, product(x_doms[i], udom, cdom), h);
        ValidatedVectorMultivariateFunctionPatch phigamma = make_function_patch({ xr,ur,tr }, phigamma_, { X[i][0],X[i][1],U[i],0,t });
        ValidatedVectorMultivariateFunctionPatch phi = project_function(phigamma, Range(0, X[i].size()));
        ValidatedScalarMultivariateFunctionPatch gamma = phigamma[X[i].size() + 1];
        ValidatedVectorMultivariateFunctionPatch psi = make_function_patch({ xr,ur }, phi, { X[i][0],X[i][1],U[i],h });

        xdom = cast_exact_box(psi.range());

        phi_patches.append(phi);
        gamma_patches.append(gamma);
    }

    assert(X.size() == U.size());
    assert(x_doms.size() == u_doms.size());
    assert(X.size() == x_doms.size() - 1);
    for (int i = 0;i < X.size();++i) { xu_restr_domains.append(X[i] | x_doms[i]); }
    for (int i = 0;i < U.size();++i) { xu_restr_domains.append(U[i] | u_doms[i]); }

    step = make_function_patch(xu_restr_domains, phi_patches[0], { X[0][0],X[0][1],U[0],h });
    state = make_function_patch(xu_restr_domains, { X[1][0],X[1][1] });
    ValidatedVectorMultivariateFunctionPatch constraint = step - state;
    for (int j = 0;j < constraint.result_size();++j) { constraints.append(constraint[j]); }

    for (int i = 1;i < num_shooting_nodes;++i)
    {
        step = make_function_patch(xu_restr_domains, phi_patches[i], { X[i][0],X[i][1],U[i],h });
        state = make_function_patch(xu_restr_domains, { X[i + 1][0],X[i + 1][1] });
        ValidatedVectorMultivariateFunctionPatch constraint = step - state;
        for (int j = 0;j < constraint.result_size();++j) { constraints.append(constraint[j]); }
    }
    g = ValidatedVectorMultivariateFunctionPatch(constraints);

    objective = make_function_patch(xu_restr_domains, gamma_patches[0], { X[0][0],X[0][1],U[0],h });
    for (int j = 1;j < num_shooting_nodes;++j)
    {
        objective += make_function_patch(xu_restr_domains, gamma_patches[j], { X[j][0],X[j][1],U[j],h });
    }

    C0BranchAndBoundOptimiser nlio(3. / 16);

    int num_constraints = g.result_size();

    List<IntervalDomainType> D_temp;
    List<IntervalDomainType> C_temp;
    for (int i = 0;i < num_constraints;++i) { C_temp.append({ -0.0_x, +0.0_x }); }

    assert(objective.result_size() == 1);
    assert(objective.domain() == g.domain());
    auto D = Box<ExactIntervalType>(objective.domain());
    auto C = Box<ExactIntervalType>(Vector<IntervalDomainType>(C_temp));

    FloatDPBoundsVector res = nlio.minimise(objective, D, g, C);

    return res;
}

void run_sim(int sim_steps, EffectiveVectorMultivariateFunction f, BoxDomainType x0, IntervalDomainType u0, BoxDomainType xdom, IntervalDomainType udom, int num_shooting_nodes, StepSizeType h)
{
    IntervalDomainType cdom = { -0.0_x, +0.0_x };
    IntervalDomainType tdom = { 0.0_x, h };
    RealVariable t("t");

    for (int i = 0;i < sim_steps;++i)
    {
        std::cout << "Simulating step ---------------\t[" << i << "]" << "\n";

        RealVectorVariable x = RealVectorVariable("x" + to_string(i), 2);
        RealVariable u = RealVariable("u" + to_string(i));

        PRINT(x0);
        PRINT(udom);
        FloatDPBoundsVector vars_opt = optimization_step(f, x0, xdom, udom, num_shooting_nodes, h);

        List<FloatDPBounds> x_opt; for (int i = 0;i < 2 * (1 + num_shooting_nodes);++i) { x_opt.append(vars_opt[i]); }
        List<FloatDPBounds> u_opt; for (int i = 2 * (1 + num_shooting_nodes);i < vars_opt.size();++i) { u_opt.append(vars_opt[i]); }

        udom = cast_exact(u_opt[0]);

        VectorVariableBoxDomainType xr = x | xdom;
        VariableIntervalDomainType ur = u | udom;
        VariableIntervalDomainType tr = t | tdom;

        ValidatedVectorMultivariateFunctionPatch PHI_GAMMA_ = integrateDynamics(f, product(xdom, udom, cdom), h);
        ValidatedVectorMultivariateFunctionPatch PHI_GAMMA = make_function_patch({ xr,ur,tr }, PHI_GAMMA_, { x[0],x[1],u,0,t });
        ValidatedVectorMultivariateFunctionPatch PHI = project_function(PHI_GAMMA, Range(0, x.size()));
        ValidatedScalarMultivariateFunctionPatch GAMMA = PHI_GAMMA[x.size() + 1];
        ValidatedVectorMultivariateFunctionPatch PSI = make_function_patch({ xr,ur }, PHI, { x[0],x[1],u,h });

        xdom = cast_exact_box(PSI.range());
        FloatDPBoundsVector xcb = xdom.centre();
        BoxRangeType xrr(xcb);
        x0 = cast_exact_box(xrr);

    }
}
int main()
{
    const int num_shooting_nodes = 5;
    const int sim_steps = 20;

    RealVectorVariable x("x", 2);
    RealVariable u("u");
    RealVariable t("t");
    RealVariable c("c");
    StepSizeType h = 0.0625_x;

    BoxDomainType x_start = { {0.25_x, 0.5_x},{0.0_x, 0.0_x} };
    BoxDomainType xdom = { {-1.0_x, +1.0_x},{-1.0_x, +1.0_x} };

    IntervalDomainType u_start = { 0.5_x, 0.5_x };
    IntervalDomainType udom = { -0.5_x, +0.5_x };

    IntervalDomainType tdom = { 0, h };

    Vector<RealExpression> x_dot = { x[1], u };
    RealExpression u_dot = 0;
    RealExpression c_dot = pow(x[0], 2) + pow(x[1], 2) + pow(u, 2);

    EffectiveVectorMultivariateFunction f_xuc = Function({ x[0], x[1], u, c }, { x_dot[0], x_dot[1], u_dot, c_dot });

    run_sim(sim_steps, f_xuc, x_start, u_start, xdom, udom, num_shooting_nodes, h);

    return 0;
};
