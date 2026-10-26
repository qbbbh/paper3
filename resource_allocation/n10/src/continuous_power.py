"""Continuous sum-of-link-EE optimization by quadratic transform and SCA.

For fixed SINR auxiliary v, A(p)=2 v sqrt(S(p))-v² gap D(p) is concave.
Introduce u>=0 and impose exp(ln(2)*u²/B_scaled)-1 <= A(p).
Then u² is a lower bound on scaled rate, without clipping a logarithm or
square root. For fixed EE auxiliary y, maximize sum(2*y*u-y²*(Pc+p)).
The subproblem has a concave objective and convex constraints in linear p,u.
SLSQP is a numerical subproblem solver; its failures are logged, not promoted
to convergence claims. Returned powers are verified against original rates.
An explicitly supplied QoS constraint is enforced in the surrogate. The eight-point solver is initialization
only; output powers range over the full closed power interval.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import minimize
from .dinkelbach_sca import _Physics

RATE_SCALE = 1e8


@dataclass
class ContinuousInfo:
    status: str
    start_mean_ee: float
    final_mean_ee: float
    history: list
    subproblems: list

    def to_dict(self):
        return asdict(self)


def surrogate(phys, anchor):
    """Return unclipped concave SINR lower bound and its [link,param] Jacobian."""
    v = phys.x(anchor) / (phys.gap * phys.D(anchor))

    def value(p):
        a = 2*v*phys.x(p) - v*v*phys.gap*phys.D(p)
        jac = (2*v[None, :]*phys.dx(p)
               - (v*v*phys.gap)[None, :]*phys.dD(p)).T
        return a, jac
    return value


def continuous_sum_ee(links, gains, ports, cfg, *, p_start,
                      max_rounds=30, tolerance=1e-7, q_min_bps=0.):
    if not links:
        return np.zeros(0), ContinuousInfo('empty', 0., 0., [], [])
    if cfg.vlc.noise_model != 'physical':
        raise ValueError('continuous SCA currently supports physical VLC noise only')
    phys = _Physics(links, gains, ports, cfg)
    p = np.asarray(p_start, float).copy()
    if p.shape != (len(links),) or not np.isfinite(p).all():
        raise ValueError('invalid continuous power start')
    p = np.clip(p, phys.lo, phys.hi)
    start = phys.mean_ee(p)
    history, logs = [start], []
    status = 'iteration_limit'
    bw = phys.bw / RATE_SCALE
    # A scalar target is broadcast; a per-link vector lets the caller raise the
    # target only on the links that are short (used by the chance-constraint
    # repair, which escalates each link by its own predicted low/mean ratio).
    q = np.asarray(q_min_bps, float)
    if q.ndim == 0:
        q = np.full(len(links), float(q))
    if q.shape != (len(links),) or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError('q_min_bps must be a nonnegative scalar or per-link vector')
    noise_floor = np.where(phys.is_rf, phys.noise_rf, phys.noise_base)
    upper_rates = phys.bw*np.log2(1+phys.x(phys.hi)**2/(phys.gap*noise_floor))
    if np.any(q > 0) and np.any(upper_rates < q):
        return p, ContinuousInfo('qos_not_found',start,start,history,
                                [{'phase':'feasibility','accepted':False,
                                  'reason':'fixed_matching_isolated_rate_upper_bound_below_q'}])
    if np.any(q > 0):
        threshold = np.expm1(np.log(2)*q/phys.bw)
        cnorm_q = np.maximum(threshold, 1e-8)
        def deficit(power):
            sinr = phys.x(power)**2/(phys.gap*phys.D(power))
            return float(np.maximum(1-sinr/threshold, 0.).sum())
        # Phase I: convex SCA slack minimization. Failure means this local search
        # did not find feasibility; it is never a proof of physical infeasibility.
        for iteration in range(40):
            if np.all(phys.rates(p) >= q*(1-1e-8)):
                break
            bound = surrogate(phys,p)
            scale = np.maximum(p,.01)
            k = phys.k
            def con(w):
                return (bound(w[:k]*scale)[0]-threshold)/cnorm_q+w[k:]
            def jac(w):
                return np.c_[bound(w[:k]*scale)[1]*scale[None,:]/cnorm_q[:,None],np.eye(k)]
            x0 = np.r_[p/scale,np.maximum((threshold-bound(p)[0])/cnorm_q,0.)]
            res = minimize(lambda w:(float(w[k:].sum()),np.r_[np.zeros(k),np.ones(k)]),
                           x0,jac=True,method='SLSQP',bounds=list(zip(phys.lo/scale,phys.hi/scale))+[(0,None)]*k,
                           constraints=[{'type':'ineq','fun':con,'jac':jac}],
                           options={'maxiter':200,'ftol':1e-10})
            cand = res.x[:k]*scale
            good = (np.isfinite(cand).all() and np.all(cand>=phys.lo-1e-12)
                    and np.all(cand<=phys.hi+1e-12) and deficit(cand)<deficit(p)-1e-10)
            logs.append({'phase':'feasibility','round':iteration,'success':bool(res.success),
                         'status':int(res.status),'nit':int(res.nit),'message':str(res.message),
                         'accepted':bool(good),'scaled_constraint_violation':max(0.,-float(con(res.x).min()))})
            if not good: break
            p = cand.copy()
        if np.any(phys.rates(p) < q*(1-1e-8)):
            return p, ContinuousInfo('qos_not_found', start, phys.mean_ee(p), history, logs)
        # Restoring QoS may lower EE; monotonic EE claims begin at this feasible
        # point, not at an infeasible unconstrained grid initialization.
        history = [phys.mean_ee(p)]
    # An interference-free upper bound is used only to bound the epigraph
    # variable; its exponent is never clipped in function/gradient evaluations.
    umax = np.sqrt(upper_rates/RATE_SCALE)
    for iteration in range(max_rounds):
        r = phys.rates(p) / RATE_SCALE
        u0 = np.sqrt(np.maximum(r, 0.))
        y = u0 / (phys.static+p)
        scale = np.maximum(p, .01)
        norm = max(float(np.sum(r/(phys.static+p))), 1e-12)
        bound = surrogate(phys, p)
        a0, _ = bound(p)
        cnorm = np.maximum(1., a0)
        k = phys.k

        def objective(w):
            power, u = w[:k]*scale, w[k:]
            value = -np.sum(2*y*u-y*y*(phys.static+power))/norm
            grad = np.r_[y*y*scale, -2*y]/norm
            return float(value), grad

        def constraint(w):
            a, _ = bound(w[:k]*scale)
            return (a-np.expm1(np.log(2)*w[k:]**2/bw))/cnorm

        def jacobian(w):
            _, jac = bound(w[:k]*scale)
            u = w[k:]
            du = -np.exp(np.log(2)*u*u/bw)*2*np.log(2)*u/bw
            return np.c_[jac*scale[None, :], np.diag(du)]/cnorm[:, None]

        res = minimize(objective, np.r_[p/scale, u0], jac=True, method='SLSQP',
                       bounds=list(zip(phys.lo/scale, phys.hi/scale))+list(zip(np.sqrt(q/RATE_SCALE), umax)),
                       constraints=[{'type':'ineq','fun':constraint,'jac':jacobian}],
                       options={'maxiter':200, 'ftol':1e-10})
        violation = max(0., -float(np.min(constraint(res.x))))
        candidate = np.asarray(res.x[:k])*scale
        finite = np.isfinite(candidate).all()
        in_box = finite and np.all(candidate >= phys.lo-1e-12) and np.all(candidate <= phys.hi+1e-12)
        value = phys.mean_ee(candidate) if in_box else float('-inf')
        accepted = bool(res.success and violation <= 1e-7 and in_box
                        and np.all(phys.rates(candidate) >= q*(1-1e-8))
                        and value > history[-1]*(1+1e-12))
        logs.append({'round':iteration, 'success':bool(res.success), 'status':int(res.status),
                     'message':str(res.message), 'nit':int(res.nit),
                     'scaled_constraint_violation':violation, 'accepted':accepted})
        if not res.success or violation > 1e-7:
            status = 'subproblem_failed'
            break
        if not accepted:
            status = 'no_improvement'
            break
        relative = (value-history[-1])/max(abs(history[-1]), 1e-12)
        p = candidate.copy()
        history.append(value)
        if relative <= tolerance:
            status = 'small_improvement'
            break
    return p, ContinuousInfo(status, start, phys.mean_ee(p), history, logs)
