"""Code for orchestrating inter-device interactions. 

This should only be relevant for developers, not users, as this code is used
under the hood when interacting devices are injected (e.g., light and opsin)."""
from __future__ import annotations

from typing import Tuple

from attrs import define, field
from brian2 import NeuronGroup, Subgroup, Synapses, defaultclock
from brian2.units.allunits import joule, kgram, meter, meter2, second
from numpy import pi

from cleo.coords import coords_from_ng
from cleo.utilities import brian_safe_name

import warnings


@define(repr=False)
class DeviceInteractionRegistry:
    """Facilitates the creation and maintenance of 'neurons' and 'synapses'
    implementing many-to-many light-opsin/indicator relationships"""

    sim: "CLSimulator" = field()

    subgroup_idx_for_light: dict["Light", slice] = field(factory=dict, init=False)
    """Maps light to its indices in the :attr:`light_source_ng`"""

    light_source_ng: NeuronGroup = field(init=False, default=None)
    """Represents ALL light sources (multiple devices)"""

    ldds_for_ng: dict[NeuronGroup, set["LightDependent"]] = field(
        factory=dict, init=False
    )
    """Maps neuron group to the light-dependent devices injected into it"""

    lights_for_ng: dict[NeuronGroup, set["Light"]] = field(factory=dict, init=False)
    """Maps neuron group to the lights injected into it"""

    light_prop_syns: dict[Tuple["LightDependent", NeuronGroup], Synapses] = field(
        factory=dict, init=False
    )
    """Maps (light-dependent device, neuron group) to the synapses implementing light propagation"""

    connections: set[Tuple["Light", "LightDependent", NeuronGroup]] = field(
        factory=set, init=False
    )
    """Set of (light, light-dependent device, neuron group) tuples representing
    previously created connections."""

    raster_fov: int = field(default=500 * 1e-6 * meter, kw_only=True)

    raster_enable: int = field(default=0, kw_only=True)

    light_prop_model = """
        T : 1
        scale : 1
        epsilon : 1
        Ephoton : joule
        raster_enable : 1
        scan_period : second
        raster_dwell_time: second
        Irr_norm = epsilon * T * Irr0_pre : watt/meter**2 
        Irr_raster = epsilon * T * Irr0_pre * int(((t + scan_period * (i/N)) % (scan_period)) < (raster_dwell_time * scale * 10)) / (scale * 10): watt/meter**2 
        Irr_post = Irr_norm * (1 - raster_enable) + Irr_raster * raster_enable : watt/meter**2 (summed)
        phi_post = Irr_post / Ephoton : 1/second/meter**2 (summed)
    """
    """Model used in light propagation synapses"""

    brian_objects: set = field(factory=set, init=False)
    """Stores all Brian objects created (and injected into the network) by this registry"""

    def register(self, device: "InterfaceDevice", ng: NeuronGroup) -> None:
        """Registers a device injection with the registry.

        Parameters
        ----------
        device : InterfaceDevice
            Device being injected
        ng : NeuronGroup
            Neurons being injected into
        """
        ancestor_classes = [t.__name__ for t in type(device).__mro__]
        if "Light" in ancestor_classes:
            self.register_light(device, ng)
        elif "LightDependent" in ancestor_classes:
            self.register_ldd(device, ng)

    def connect_light_to_ldd_for_ng(
        self, light: "Light", ldd: "LightDependent", ng: NeuronGroup
    ) -> None:
        """Connects a light to a light-dependent device for a given neuron group.

        Parameters
        ----------
        light : Light
            Light being injected
        ldd : LightDependent
            Light-dependent device the light will affect
        ng : NeuronGroup
            Neurons affected by the light-dependent device

        Raises
        ------
        ValueError
            if the connection has already been made
        """
        epsilon = ldd.epsilon(light.wavelength)
        if epsilon == 0:
            return

        light_prop_syn = self._get_or_create_light_prop_syn(ldd, ng)
        if (light, ldd, ng) in self.connections:
            raise ValueError(f"{light} already connected to {ldd.name} for {ng.name}")

        i_source = self.subgroup_idx_for_light[light]
        light_prop_syn.epsilon[i_source, :] = epsilon
        light_prop_syn.T[i_source, :] = light.transmittance(coords_from_ng(ng)).ravel()
        light_prop_syn.scan_period = second / light.scan_freq
        light_prop_syn.raster_dwell_time = (pi * 10 ** -10 * meter2)/ (pi*(self.raster_fov/2)**2) * second / light.scan_freq
        light_prop_syn.scale = 1 + (defaultclock.dt > light_prop_syn.raster_dwell_time).astype(int) * (defaultclock.dt / (light_prop_syn.raster_dwell_time) - 1)
        light_prop_syn.raster_enable = self.raster_enable
        # fmt: off
        # Ephoton = h*c/lambda
        light_prop_syn.Ephoton[i_source, :] = (
            6.63e-34 * meter2 * kgram / second
            * 2.998e8 * meter / second
            / light.wavelength
        )
        # fmt: on
        self.connections.add((light, ldd, ng))

    def _add_brian_object(self, obj):
        self.brian_objects.add(obj)
        self.sim.network.add(obj)

    def _remove_brian_object(self, obj):
        self.brian_objects.remove(obj)
        self.sim.network.remove(obj)

    def _get_or_create_light_prop_syn(
        self, ldd: "LightDependent", ng: NeuronGroup
    ) -> Synapses:
        if (ldd, ng) not in self.light_prop_syns:
            light_agg_ng = ldd.light_agg_ngs[ng.name]

            light_prop_syn = Synapses(
                self.light_source_ng,
                light_agg_ng,
                model=self.light_prop_model,
                name=f"light_prop_{brian_safe_name(ldd.name)}_{ng.name}",
            )
            light_prop_syn.connect()
            # non-zero initialization to avoid nans from /0
            light_prop_syn.Ephoton = 1 * joule
            self._add_brian_object(light_prop_syn)
            self.light_prop_syns[(ldd, ng)] = light_prop_syn

        return self.light_prop_syns[(ldd, ng)]

    def register_ldd(self, ldd: "LightDependent", ng: NeuronGroup):
        """Connects lights previously injected into this neuron group to this light-dependent device"""
        if ng not in self.ldds_for_ng:
            self.ldds_for_ng[ng] = set()
        self.ldds_for_ng[ng].add(ldd)
        prev_injct_lights = self.lights_for_ng.get(ng, set())
        for light in prev_injct_lights:
            self.connect_light_to_ldd_for_ng(light, ldd, ng)

    def init_register_light(self, light: "Light") -> Subgroup:
        """Creates neurons for the light source, if they don't already exist"""
        if self.light_source_ng is not None:
            Irr0_prev = self.light_source_ng.Irr0
            n_prev = self.light_source_ng.N
            # need to remove the old light source from the network
            self._remove_brian_object(self.light_source_ng)
        else:
            Irr0_prev = []
            n_prev = 0

        # create new one
        self.light_source_ng = NeuronGroup(
            n_prev + light.n, "Irr0: watt/meter**2", name="light_source"
        )
        if n_prev > 0:
            self.light_source_ng[:n_prev].Irr0 = Irr0_prev
        self._add_brian_object(self.light_source_ng)
        self.subgroup_idx_for_light[light] = slice(n_prev, n_prev + light.n)

        # remove and replace light_prop_syns for previous connections
        for light_prop_syn in self.light_prop_syns.values():
            self._remove_brian_object(light_prop_syn)
        prev_cxns = self.connections.copy()
        self.connections.clear()
        self.light_prop_syns.clear()
        for light, ldd, ng in prev_cxns:
            self.connect_light_to_ldd_for_ng(light, ldd, ng)
        assert prev_cxns == self.connections

    def register_light(self, light: "Light", ng: NeuronGroup):
        """Connects light to light-dependent devices already injected into this neuron group"""
        # create new connections for this light
        if ng not in self.lights_for_ng:
            self.lights_for_ng[ng] = set()
        self.lights_for_ng[ng].add(light)
        prev_injct_ldds = self.ldds_for_ng.get(ng, set())
        for ldd in prev_injct_ldds:
            self.connect_light_to_ldd_for_ng(light, ldd, ng)

    def source_for_light(self, light: "Light") -> Subgroup:
        """Returns the subgroup representing the given light source"""
        i = self.subgroup_idx_for_light[light]
        return self.light_source_ng[i]
    
    def update_fov(self, new_fov: float):
        self.raster_enable = 1
        def custom_warning_format(message, category, filename, lineno, line=None):
            return f"{category.__name__}: {message}\n"
        warnings.formatwarning = custom_warning_format
        if self.raster_fov != new_fov:
            self.raster_fov = new_fov
            warnings.warn("Warning: cleo currently supports use of single scanning fov in simulation, the latest update parameter will be used", UserWarning)


registries: dict["CLSimulator", DeviceInteractionRegistry] = {}
"""Maps simulator to its registry"""


def registry_for_sim(sim: "CLSimulator") -> DeviceInteractionRegistry:
    """Returns the registry for the given simulator"""
    assert sim is not None
    if sim not in registries:
        registries[sim] = DeviceInteractionRegistry(sim)
    return registries[sim]
