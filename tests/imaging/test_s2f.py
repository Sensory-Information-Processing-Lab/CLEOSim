import pytest
from brian2 import NeuronGroup, ms, np, mV
from cleo.imaging.s2f import (
    S2FGECI,
    S2FLightDependentGECI,
    geci_s2f,
)


def test_s2f_init():
    model = S2FGECI(
        name="test_geci",
        rise=16.0 * ms,
        decay1=95.0 * ms,
        decay2=400.0 * ms,
        r=0.2,
        Fm=1.0,
        Ca0=0.1,
        beta=0.2,
        F0=0.0,
        sigma_noise=0.01,
    )
    assert model.name == "test_geci"


def test_s2f_light_dependent_init():
    model = S2FLightDependentGECI(
        name="test_light",
        rise=16.0 * ms,
        decay1=95.0 * ms,
        decay2=400.0 * ms,
        r=0.2,
        Fm=1.0,
        Ca0=0.1,
        beta=0.2,
        F0=0.0,
        sigma_noise=0.01,
        spectrum=[(500.0, 1.2)],
    )
    assert model.name == "test_light"
    assert model.spectrum == [(500.0, 1.2)]


def test_geci_s2f_factory():
    # Should return a regular S2FGECI
    model_regular = geci_s2f(
        "regular_test", light_dependent=False, rise=5.0 * ms, decay1=10.0 * ms
    )
    assert isinstance(model_regular, S2FGECI)
    assert not isinstance(model_regular, S2FLightDependentGECI)

    # Should return a light-dependent S2FLightDependentGECI
    model_light = geci_s2f(
        "light_test",
        light_dependent=True,
        rise=5.0 * ms,
        decay1=10.0 * ms,
        spectrum=[(510.0, 2.0)],
    )
    assert isinstance(model_light, S2FLightDependentGECI)


@pytest.mark.parametrize(
    "rise,decay1,decay2,r", [(16.0, 95.0, 400.0, 0.2), (50, 1700, 0, 0)]
)
def test_s2f_calcium_response(rise, decay1, decay2, r):
    model = S2FGECI(
        name="test_params",
        rise=rise * ms,
        decay1=decay1 * ms,
        decay2=decay2 * ms,
        r=r,
        Fm=1.0,
        Ca0=0.0,
        beta=1.0,
        F0=0.0,
        sigma_noise=0.0,
    )

    # Mock the network time and spikes
    class MockSim:
        def __init__(self, t):
            self.network = self
            self.t = t

    model.sim = MockSim(t=50.0 * ms)
    spike_times = np.array([45.0, 30.0]) * ms
    response = model.spike_to_calcium(spike_times)
    assert len(response) == 2
    # Check simple positivity or similar
    assert np.all(response >= 0)


def test_connect_to_neuron_group():
    from brian2 import Synapses

    # Minimal mock network
    ng = NeuronGroup(10, "dv/dt = 0*Hz : 1", threshold=0 * mV, name="test_ng")
    model = S2FGECI(
        name="test_connect",
        rise=16.0 * ms,
        decay1=95.0 * ms,
        decay2=400.0 * ms,
        r=0.2,
        Fm=1.0,
        Ca0=0.1,
        beta=0.2,
        F0=0.0,
        sigma_noise=0.01,
    )
    model.connect_to_neuron_group(ng)
    assert "test_ng" in model.spike_monitors
