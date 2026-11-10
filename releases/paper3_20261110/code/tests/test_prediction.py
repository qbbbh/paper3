import copy
import pickle
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from dataclasses import replace
from scipy.special import exp1
from p3base import channel
from p3base.config import FoundationConfig
from p3base.estimator import ChannelEstimator
from p3base.block import GeometryContext,predict_block


def setup():
    cfg=FoundationConfig()
    ctx=GeometryContext(2,cfg.vehicle,np.empty((0,6)),np.full(2,4.5),np.full(2,1.8),np.full(2,1.5))
    state=dict(center_xy_m=np.array([[0.,0.],[12.,0.]]),heading_rad=np.zeros(2),speed_mps=np.ones(2))
    est=ChannelEstimator(2,cfg)
    ports=dict(tx=np.zeros((2,2),int),rx=np.ones((2,2),int))
    return est,state,ctx,ports


def test_predictor_no_state_mutation_and_reproducible():
    est,state,ctx,ports=setup()
    before=pickle.dumps(est)
    a=predict_block(est,state,ctx,[(0,1,'RF')],[.1],ports,0.,draws=8,seed=3)
    b=predict_block(est,state,ctx,[(0,1,'RF')],[.1],ports,0.,draws=8,seed=3)
    assert pickle.dumps(est)==before
    np.testing.assert_array_equal(a['block_rate_samples_bps'],b['block_rate_samples_bps'])
    assert np.isfinite(a['block_rate_samples_bps']).all()
    assert not hasattr(ctx,'at') and not hasattr(ctx,'time_s')


def test_future_posterior_rejected():
    est,state,ctx,ports=setup()
    est.state_time['rf']=.1
    with pytest.raises(ValueError):
        predict_block(est,state,ctx,[(0,1,'RF')],[.1],ports,0.,draws=2)


def test_observation_cutoff():
    est,_,_,_=setup()
    class Provider:
        def __init__(self): self.seen=[]
        def rf(self,t):
            assert t+.002<=.1+1e-12
            self.seen.append(t)
            return np.ones((2,2),complex),np.full((2,2),.1),np.ones((2,2),bool)
        def vlc(self,t):
            assert t+.005<=.1+1e-12
            return np.zeros((2,2,2,2)),np.ones((2,2,2,2)),np.ones((2,2,2,2),bool)
    p=Provider();est.advance(.1,p,rf_tau=np.full((2,2),.03),rf_prior_var=np.ones((2,2)))
    assert np.isclose(max(p.seen),.098)


def test_rayleigh_average_rate_against_analytic_integral():
    # Independent closed-form E[log(1+c Z)], Z~Exp(1), fixed isolated RF link.
    # Checks that temporal sampling and rate integration have the right mean.
    est,state,ctx,ports=setup()
    cfg=FoundationConfig()
    cfg=cfg.with_(estimation=replace(cfg.estimation,rf_state='complex_ar1'))
    est=ChannelEstimator(2,cfg)
    state['speed_mps'][:]=0
    gain=channel.rf_geometry(ctx,state,cfg.rf)['gain'][0,1]
    c=cfg.rf.pa_efficiency*.1*gain/(cfg.rf.noise_w*cfg.rf.gap)
    expected=cfg.rf.bandwidth_hz/np.log(2)*np.exp(1/c)*exp1(1/c)
    result=predict_block(est,state,ctx,[(0,1,'RF')],[.1],ports,0.,draws=2048,seed=22)
    assert abs(result['mean_rate_bps'][0]/expected-1)<.04
