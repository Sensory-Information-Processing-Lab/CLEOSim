"""Contains opsin model, parameters, and OptogeneticIntervention device"""
from __future__ import annotations
from typing import Tuple, Any

from brian2 import Synapses, NeuronGroup
from brian2.units import *
from brian2.units.allunits import meter2
import brian2.units.unitsafefunctions as usf
from brian2.core.base import BrianObjectException
import numpy as np
import matplotlib
from matplotlib import colors
from matplotlib.artist import Artist
from matplotlib.collections import PathCollection

from cleosim.utilities import wavelength_to_rgb
from cleosim.stimulators import Stimulator


four_state = """
    dC1/dt = Gd1*O1 + Gr0*C2 - Ga1*C1 : 1 (clock-driven)
    dO1/dt = Ga1*C1 + Gb*O2 - (Gd1+Gf)*O1 : 1 (clock-driven)
    dO2/dt = Ga2*C2 + Gf*O1 - (Gd2+Gb)*O2 : 1 (clock-driven)
    C2 = 1 - C1 - O1 - O2 : 1
    # dC2/dt = Gd2*O2 - (Gr0+Ga2)*C2 : 1 (clock-driven)

    Theta = int(phi > 0*phi) : 1
    Hp = Theta * phi**p/(phi**p + phim**p) : 1
    Ga1 = k1*Hp : hertz
    Ga2 = k2*Hp : hertz
    Hq = Theta * phi**q/(phi**q + phim**q) : 1
    Gf = kf*Hq + Gf0 : hertz
    Gb = kb*Hq + Gb0 : hertz

    fphi = O1 + gamma*O2 : 1
    fv = (1 - exp(-(V_VAR_NAME_post-E)/v0)) / -2 : 1

    IOPTO_VAR_NAME_post = g0*fphi*fv*(V_VAR_NAME_post-E)*rho_rel : ampere (summed)
    rho_rel : 1
"""
"""Equations for the 4-state model from PyRhO (Evans et al. 2016).

Assuming this model is defined on "synapses" influencing a post-synaptic
target population. rho_rel is channel density relative to standard model fit;
modifying it post-injection allows for heterogeneous opsin expression.

IOPTO_VAR_NAME and V_VAR_NAME are substituted on injection
"""

ChR2_four_state = {
    "g0": 114000 * psiemens,
    "gamma": 0.00742,
    "phim": 2.33e17 / mm2 / second,  # *photon, not in Brian2
    "k1": 4.15 / ms,
    "k2": 0.868 / ms,
    "p": 0.833,
    "Gf0": 0.0373 / ms,
    "kf": 0.0581 / ms,
    "Gb0": 0.0161 / ms,
    "kb": 0.063 / ms,
    "q": 1.94,
    "Gd1": 0.105 / ms,
    "Gd2": 0.0138 / ms,
    "Gr0": 0.00033 / ms,
    "E": 0 * mV,
    "v0": 43 * mV,
    "v1": 17.1 * mV,
}
"""Parameters for the 4-state ChR2 model.

Taken from try.projectpyrho.org's default 4-state params.
"""

default_blue = {
    "R0": 0.1 * mm,  # optical fiber radius
    "NAfib": 0.37,  # optical fiber numerical aperture
    "wavelength": 473 * nmeter,
    # NOTE: the following depend on wavelength and tissue properties and thus would be different for another wavelength
    "K": 0.125 / mm,  # absorbance coefficient
    "S": 7.37 / mm,  # scattering coefficient
    "ntis": 1.36,  # tissue index of refraction
}
"""Light parameters for 473 nm wavelength delivered via an optic fiber.

From Foutz et al., 2012"""


class OptogeneticIntervention(Stimulator):
    """Enables optogenetic stimulation of the network.

    Essentially "transfects" neurons and provides a light source

    Requires neurons to have 3D spatial coordinates already assigned.
    Will deliver current via a Brian :class:`~brian2.synapses.synapses.Synapses`
    object.

    See :meth:`connect_to_neuron_group` for optional keyword parameters
    that can be specified when calling
    :meth:`cleosim.CLSimulator.inject_stimulator`.
    """

    opto_syns: dict[str, Synapses]
    """Stores the synapse objects implementing the opsin model,
    with NeuronGroup name keys and Synapse values."""

    max_Irr0_mW_per_mm2: float
    """The maximum irradiance the light source can emit.
    
    Usually determined by hardware in a real experiment."""

    max_Irr0_mW_per_mm2_viz: float
    """Maximum irradiance for visualization purposes. 
    
    i.e., the level at or above which the light appears maximally bright.
    Only relevant in video visualization.
    """

    def __init__(
        self,
        name: str,
        opsin_model: str,
        opsin_params: dict,
        light_model_params: dict,
        location: Quantity = (0, 0, 0) * mm,
        direction: Tuple[float, float, float] = (0, 0, 1),
        max_Irr0_mW_per_mm2: float = None,
    ):
        """
        Parameters
        ----------
        name : str
            Unique identifier for stimulator
        opsin_model : str
            Brian equation string to use for the opsin model.
            See :attr:`four_state` for an example.
        opsin_params : dict
            Parameters in the form of a namespace dict for the Brian equations.
            See :attr:`ChR2_four_state` for an example.
        light_model_params : dict
            Parameters for the light propagation model in Foutz et al., 2012.
            See :attr:`default_blue` for an example.
        location : Quantity, optional
            (x, y, z) coords with Brian unit specifying where to place
            the base of the light source, by default (0, 0, 0)*mm
        direction : Tuple[float, float, float], optional
            (x, y, z) vector specifying direction in which light
            source is pointing, by default (0, 0, 1)
        """
        super().__init__(name, 0)
        self.opsin_model = opsin_model
        self.opsin_params = opsin_params
        self.light_model_params = light_model_params
        self.location = location
        # direction unit vector
        self.dir_uvec = (direction / np.linalg.norm(direction)).reshape((3, 1))
        self.opto_syns = {}
        self.max_Irr0_mW_per_mm2 = max_Irr0_mW_per_mm2
        self.max_Irr0_mW_per_mm2_viz = None

    def _Foutz12_transmittance(self, r, z, scatter=True, spread=True, gaussian=True):
        """Foutz et al. 2012 transmittance model: Gaussian cone with Kubelka-Munk propagation"""

        if spread:
            # divergence half-angle of cone
            theta_div = np.arcsin(
                self.light_model_params["NAfib"] / self.light_model_params["ntis"]
            )
            Rz = self.light_model_params["R0"] + z * np.tan(
                theta_div
            )  # radius as light spreads("apparent radius" from original code)
            C = (self.light_model_params["R0"] / Rz) ** 2
        else:
            Rz = self.light_model_params["R0"]  # "apparent radius"
            C = 1

        if gaussian:
            G = 1 / np.sqrt(2 * np.pi) * np.exp(-2 * (r / Rz) ** 2)
        else:
            G = 1

        def kubelka_munk(dist):
            S = self.light_model_params["S"]
            a = 1 + self.light_model_params["K"] / S
            b = np.sqrt(a ** 2 - 1)
            dist = np.sqrt(r ** 2 + z ** 2)
            return b / (a * np.sinh(b * S * dist) + b * np.cosh(b * S * dist))

        M = kubelka_munk(np.sqrt(r ** 2 + z ** 2)) if scatter else 1

        T = G * C * M
        return T

    def _get_rz_for_xyz(self, x, y, z):
        """Assumes x, y, z already have units"""

        def flatten_if_needed(var):
            if len(var.shape) != 1:
                return var.flatten()
            else:
                return var

        # have to add unit back on since it's stripped by vstack
        coords = (
            np.vstack(
                [flatten_if_needed(x), flatten_if_needed(y), flatten_if_needed(z)]
            ).T
            * meter
        )
        rel_coords = coords - self.location  # relative to fiber location
        # must use brian2's dot function for matrix multiply to preserve
        # units correctly.
        zc = usf.dot(rel_coords, self.dir_uvec)  # distance along cylinder axis
        # just need length (norm) of radius vectors
        # not using np.linalg.norm because it strips units
        r = np.sqrt(np.sum((rel_coords - usf.dot(zc, self.dir_uvec.T)) ** 2, axis=1))
        r = r.reshape((-1, 1))
        return r, zc

    def connect_to_neuron_group(
        self, neuron_group: NeuronGroup, **kwparams: Any
    ) -> None:
        """Configure opsin and light source to stimulate given neuron group.

        Parameters
        ----------
        neuron_group : NeuronGroup
            The neuron group to stimulate with the given opsin and light source

        Keyword args
        ------------
        p_expression : float
            Probability (0 <= p <= 1) that a given neuron in the group
            will express the opsin. 1 by default.
        rho_rel : float
            The expression level, relative to the standard model fit,
            of the opsin. 1 by default. For heterogeneous expression,
            this would have to be modified in the opsin synapse post-injection,
            e.g., ``opto.opto_syns["neuron_group_name"].rho_rel = ...``
        Iopto_var_name : str
            The name of the variable in the neuron group model representing
            current from the opsin
        v_var_name : str
            The name of the variable in the neuron group model representing
            membrane potential
        """
        p_expression = kwparams.get("p_expression", 1)
        Iopto_var_name = kwparams.get("Iopto_var_name", "Iopto")
        v_var_name = kwparams.get("v_var_name", "v")
        for variable, unit in zip([v_var_name, Iopto_var_name], [volt, amp]):
            if (
                variable not in neuron_group.variables
                or neuron_group.variables[variable].unit != unit
            ):
                raise BrianObjectException(
                    (
                        f"{variable} : {unit.name} needed in the model of NeuronGroup"
                        f"{neuron_group.name} to connect OptogeneticIntervention."
                    ),
                    neuron_group,
                )
        # opsin synapse model needs modified names
        modified_opsin_model = self.opsin_model.replace(
            "IOPTO_VAR_NAME", Iopto_var_name
        ).replace("V_VAR_NAME", v_var_name)

        # fmt: off
        # Ephoton = h*c/lambda
        E_photon = (
            6.63e-34 * meter2 * kgram / second
            * 2.998e8 * meter / second
            / self.light_model_params["wavelength"]
        )
        # fmt: on

        light_model = """
            Irr = Irr0*T : watt/meter**2
            Irr0 : watt/meter**2 
            T : 1
            phi = Irr / Ephoton : 1/second/meter**2
        """

        opto_syn = Synapses(
            neuron_group,
            model=modified_opsin_model + light_model,
            namespace=self.opsin_params,
            name=f"synapses_{self.name}_{neuron_group.name}",
            method="rk2",
        )
        opto_syn.namespace["Ephoton"] = E_photon

        if p_expression == 1:
            opto_syn.connect(j="i")
        else:
            opto_syn.connect(condition="i==j", p=p_expression)

        self._init_opto_syn_vars(opto_syn)

        # relative channel density
        opto_syn.rho_rel = kwparams.get("rho_rel", 1)
        # calculate transmittance coefficient for each point
        r, z = self._get_rz_for_xyz(neuron_group.x, neuron_group.y, neuron_group.z)
        T = self._Foutz12_transmittance(r, z).flatten()
        # reduce to subset expressing opsin before assigning
        T = [T[k] for k in opto_syn.i]

        opto_syn.T = T

        self.opto_syns[neuron_group.name] = opto_syn
        self.brian_objects.add(opto_syn)

    def add_self_to_plot(self, ax, axis_scale_unit) -> PathCollection:
        # show light with point field, assigning r and z coordinates
        # to all points
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()
        x = np.linspace(xlim[0], xlim[1], 50)
        y = np.linspace(ylim[0], ylim[1], 50)
        z = np.linspace(zlim[0], zlim[1], 50)
        x, y, z = np.meshgrid(x, y, z) * axis_scale_unit

        r, zc = self._get_rz_for_xyz(x, y, z)
        T = self._Foutz12_transmittance(r, zc)
        # filter out points with <0.001 transmittance to make plotting faster
        plot_threshold = 0.001
        idx_to_plot = T[:, 0] >= plot_threshold
        x = x.flatten()[idx_to_plot]
        y = y.flatten()[idx_to_plot]
        z = z.flatten()[idx_to_plot]
        T = T[idx_to_plot, 0]
        point_cloud = ax.scatter(
            x / axis_scale_unit,
            y / axis_scale_unit,
            z / axis_scale_unit,
            c=T,
            cmap=self._alpha_cmap_for_wavelength(),
            marker=",",
            edgecolors="none",
            label=self.name,
        )
        handles = ax.get_legend().legendHandles
        c = wavelength_to_rgb(self.light_model_params["wavelength"] / nmeter)
        opto_patch = matplotlib.patches.Patch(color=c, label=self.name)
        handles.append(opto_patch)
        ax.legend(handles=handles)
        return [point_cloud]

    def update_artists(
        self, artists: list[Artist], value, *args, **kwargs
    ) -> list[Artist]:
        self._prev_value = getattr(self, "_prev_value", None)
        if value == self._prev_value:
            return []

        assert len(artists) == 1
        point_cloud = artists[0]

        if self.max_Irr0_mW_per_mm2_viz is not None:
            max_Irr0 = self.max_Irr0_mW_per_mm2_viz
        elif self.max_Irr0_mW_per_mm2 is not None:
            max_Irr0 = self.max_Irr0_mW_per_mm2
        else:
            raise Exception(
                f"OptogeneticIntervention '{self.name}' needs max_Irr0_mW_per_mm2_viz "
                "or max_Irr0_mW_per_mm2 "
                "set to visualize light intensity."
            )

        intensity = value / max_Irr0 if value <= max_Irr0 else max_Irr0
        point_cloud.set_cmap(self._alpha_cmap_for_wavelength(intensity))
        return [point_cloud]

    def update(self, Irr0_mW_per_mm2: float):
        """Set the light intensity, in mW/mm2 (without unit)

        Parameters
        ----------
        Irr0_mW_per_mm2 : float
            Desired light intensity for light source

        Raises
        ------
        ValueError
            When intensity is negative
        """
        if Irr0_mW_per_mm2 < 0:
            raise ValueError(f"{self.name}: light intensity Irr0 must be nonnegative")
        if (
            self.max_Irr0_mW_per_mm2 is not None
            and Irr0_mW_per_mm2 > self.max_Irr0_mW_per_mm2
        ):
            Irr0_mW_per_mm2 = self.max_Irr0_mW_per_mm2
        super().update(Irr0_mW_per_mm2)
        for opto_syn in self.opto_syns.values():
            opto_syn.Irr0 = Irr0_mW_per_mm2 * mwatt / mm2

    def _init_opto_syn_vars(self, opto_syn):
        for varname, value in {"Irr0": 0, "C1": 1, "O1": 0, "O2": 0}.items():
            setattr(opto_syn, varname, value)

    def reset(self, **kwargs):
        for opto_syn in self.opto_syns.values():
            self._init_opto_syn_vars(opto_syn)

    def _alpha_cmap_for_wavelength(self, intensity=0.5):
        c = wavelength_to_rgb(self.light_model_params["wavelength"] / nmeter)
        c_clear = (*c, 0)
        c_opaque = (*c, 0.6 * intensity)
        return colors.LinearSegmentedColormap.from_list(
            "incr_alpha", [(0, c_clear), (1, c_opaque)]
        )
