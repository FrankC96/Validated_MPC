using namespace Ariadne;

void pprint(std::string str, auto var)
{
    std::cout << str << " => \n" << var << "\n";
    for (int i = 0;i < 50;++i) { std::cout << "-"; };
    std::cout << "\n";
};
#define PRINT(expr) { pprint(#expr,(expr)); }


ValidatedVectorMultivariateFunctionPatch integrateDynamics(EffectiveVectorMultivariateFunction f, ExactBoxType x_dom, StepSizeType h)
{
    auto integrator = TaylorPicardIntegrator(1E-2);

    ValidatedVectorMultivariateFunctionPatch x_d = integrator.flow_step(f, x_dom, h);
    return x_d;
};
ValidatedVectorMultivariateFunctionPatch project_function(ValidatedVectorMultivariateFunctionPatch f, Range p)
{
    return Array<ValidatedScalarMultivariateFunctionPatch>(p.size(), [&f, &p](SizeType i) { return f[p[i]]; });
};

typedef Variant<RealVectorVariable, RealVariable> RealOrVectorVariable;

class VariableIntervalDomainType : public VariableInterval<FloatDP> {
    using VariableInterval<FloatDP>::VariableInterval;
};
VariableIntervalDomainType operator| (RealVariable v, IntervalDomainType d) {
    return VariableIntervalDomainType(v, d);
}

class VectorVariableBoxDomainType {
    RealVectorVariable _var;
    BoxDomainType _dom;
public:
    RealVectorVariable variable() const { return _var; }
    BoxDomainType set() const { return _dom; }
    VectorVariableBoxDomainType(RealVectorVariable v, BoxDomainType d) : _var(v), _dom(d) {};
    VariableIntervalDomainType operator[](SizeType i) const { return _var[i] | _dom[i]; }
    friend OutputStream& operator<<(OutputStream& os, VectorVariableBoxDomainType const& vd) {
        return os << vd._var << '|' << vd._dom;
    }
};
VectorVariableBoxDomainType operator| (RealVectorVariable vv, BoxDomainType d) {
    return VectorVariableBoxDomainType(vv, d);
}

typedef Variant<VariableIntervalDomainType, VectorVariableBoxDomainType> VariableIntervalOrBoxDomainType;
typedef Variant<RealVariable, RealVectorVariable> ScalarOrVectorVariable;
typedef Variant<BoxDomainType, IntervalDomainType> BoxOrIntervalDomain;
template<class I> Box<I> make_box(InitializerList<Interval<RawFloatDP>> lst) { return Box<I>(Vector<I>(lst)); }

ValidatedScalarMultivariateFunctionPatch make_function_patch(List<VariableIntervalOrBoxDomainType> sv_arg_doms, RealExpression e) {
    List<VariableIntervalDomainType> arg_doms;
    List<RealVariable> args; for (auto arg_dom : arg_doms) { args.append(arg_dom.variable()); }
    RealSpace spc(args);
    List<IntervalDomainType> doms; for (auto arg_dom : arg_doms) { doms.append(arg_dom.interval()); }
    BoxDomainType dom = BoxDomainType(Array<IntervalDomainType>(doms.begin(), doms.end()));
    EffectiveScalarMultivariateFunction fe = make_function(spc, e);
    ValidatedScalarMultivariateTaylorFunctionModelDP fem(dom, fe, ThresholdSweeperDP(dp, 1e-8));
    return fem;
}

template<class... TS> OutputStream& operator<<(OutputStream& os, Variant<TS...> const& var) {
    std::visit([&os](auto const& t) {os << t;}, var); return os;
}
template<class T> OutputStream& operator<<(OutputStream& os, InitializerList<T> const& lst) {
    return os << List<T>(lst);
}

List<VariableIntervalDomainType> make_scalar_variable_domains(List<VariableIntervalOrBoxDomainType> const& sv_arg_doms) {
    List<VariableIntervalDomainType> arg_doms;
    for (auto sv_arg_dom : sv_arg_doms) {
        if (std::holds_alternative<VariableIntervalDomainType>(sv_arg_dom)) {
            arg_doms.append(std::get<VariableIntervalDomainType>(sv_arg_dom));
        }
        else {
            auto bx_arg_dom = std::get<VectorVariableBoxDomainType>(sv_arg_dom);
            for (SizeType i = 0; i != bx_arg_dom.variable().size(); ++i) {
                arg_doms.append(bx_arg_dom[i]);
            }
        }
    }
    return arg_doms;
}

ValidatedVectorMultivariateFunctionPatch make_function_patch(List<VariableIntervalOrBoxDomainType> sv_arg_doms, List<RealExpression> es) {
    List<VariableIntervalDomainType> arg_doms = make_scalar_variable_domains(sv_arg_doms);
    List<RealVariable> args; for (auto arg_dom : arg_doms) { args.append(arg_dom.variable()); }
    RealSpace spc(args);
    List<IntervalDomainType> doms; for (auto arg_dom : arg_doms) { doms.append(arg_dom.interval()); }
    BoxDomainType dom = BoxDomainType(Array<IntervalDomainType>(doms.begin(), doms.end()));
    Vector<RealExpression> e(es);
    EffectiveVectorMultivariateFunction fe = make_function(spc, e);
    ValidatedVectorMultivariateTaylorFunctionModelDP fem(dom, fe, ThresholdSweeperDP(dp, 1e-8));
    return fem;
}

ValidatedScalarMultivariateFunctionPatch make_function_patch(List<VariableIntervalOrBoxDomainType> sv_arg_doms, ValidatedScalarMultivariateFunctionPatch f, List<RealExpression> es) {
    return compose(cast_unrestricted(f), make_function_patch(sv_arg_doms, es));
}

ValidatedVectorMultivariateFunctionPatch make_function_patch(List<VariableIntervalOrBoxDomainType> sv_arg_doms, ValidatedVectorMultivariateFunctionPatch f, List<RealExpression> es) {
    return compose(cast_unrestricted(f), make_function_patch(sv_arg_doms, es));
}
