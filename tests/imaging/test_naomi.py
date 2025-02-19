import brian2 as b2
import pytest
from brian2 import np

import cleo
from cleo.imaging import gcamp6f_naomi


@pytest.mark.slow
def test_geci(rand_seed):
    rng = np.random.default_rng(rand_seed)
    # random spikes during first 40 ms
    n_nrns = 5
    i_spike_bound = 3
    n_spk = n_nrns * 2
    i_spk = rng.choice(range(i_spike_bound), n_spk)
    t_spk = 40 * rng.random((n_spk,)) * b2.ms
    sgg = b2.SpikeGeneratorGroup(n_nrns, i_spk, t_spk)
    sgg._N = n_nrns  # hack for assign_coords to work
    # cleo.coords.assign_coords(sgg, c_mm[:, 0], c_mm[:, 1], c_mm[:, 2])
    geci = gcamp6f_naomi()

    sim = cleo.CLSimulator(b2.Network(sgg))
    sim.inject(geci, sgg)
    dFF = geci.get_state()[sgg.name]
    assert np.shape(dFF) == (n_nrns,)
    assert np.all(dFF == 0)  # resting state, since I correct to 0
    sim.run(50 * b2.ms)
    for i_spiking_nrn in range(i_spike_bound):
        for i_nonspiking_nrn in range(i_spike_bound, n_nrns):
            assert dFF[i_spiking_nrn] > dFF[i_nonspiking_nrn]


def test_remove_sensor():
    ng = b2.NeuronGroup(1, "dv/dt = -v / (10*ms) : 1", threshold="v>1", reset="v=0")
    sensor = gcamp6f_naomi()
    spmon = b2.SpikeMonitor(ng)
    sim = cleo.CLSimulator(b2.Network(ng, spmon))
    sim.inject(sensor, ng)
    sim._remove(sensor)
    assert ng in sim.network.objects
    assert spmon in sim.network.objects
    assert sensor.synapses[ng.name] not in sim.network.objects


if __name__ == "__main__":
    pytest.main(["-sx", __file__])
