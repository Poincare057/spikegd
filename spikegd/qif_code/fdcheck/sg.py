"""Thin wrappers around Klos & Memmesheimer's spikegd (https://github.com/chklos/spikegd), in float64.
spikegd is an independent JAX implementation of the same QIF neuron (tau_m = 2 tau_s = 1); its gradients come from JAX autodiff
through the event-based simulation, so they are a reference that shares no code with our closed forms."""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from spikegd.qif import QIFNeuron

EPS = 1e-10
neuron = QIFNeuron(tau=1.0, eps=EPS)

def flowV(V, b, t): return float(neuron.flow_V(jnp.array(V), jnp.array(b), jnp.array(t)))
def tsp(V, b): return float(neuron.dt_spike(jnp.array([[V], [b]]))[0])
dflowV = jax.grad(lambda V, b, t: neuron.flow_V(V, b, t), argnums=(0, 1, 2))
dtsp = jax.grad(lambda V, b: neuron.dt_spike(jnp.stack([V, b])[:, None])[0], argnums=(0, 1))

def run(x0, W_net, W_in, t_in, idx_in, T, K):
    """x0: (2,N) initial (V,I); W_net: (N,N); W_in: (N,Nin); input spikes (t_in, idx_in). Returns spikegd's event output."""
    config = {'T': T, 'K': K}
    return neuron.event(jnp.asarray(x0), jnp.asarray(W_net), jnp.asarray(W_in), (jnp.asarray(t_in), jnp.asarray(idx_in)), config)

def output_times(out, n, k):
    """First k output spike times of neuron n (padded with nan)."""
    ts, is_in, js, _ = out
    mask = (~is_in) & (js == n)
    order = jnp.argsort(jnp.where(mask, ts, jnp.inf))
    return jnp.where(jnp.arange(k) < mask.sum(), ts[order][:k], jnp.nan)

def state_at_end(out):
    ts, is_in, js, xs = out
    return xs[jnp.sum(js >= 0)]                       # state at the trial end (spikegd convention)
