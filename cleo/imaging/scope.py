from __future__ import annotations

import warnings
from datetime import datetime
from typing import Callable

import matplotlib as mpl
import neo
import quantities as pq
from attrs import define, field, fields
from brian2 import NeuronGroup, Quantity, Unit, meter, mm, ms, np, um
from jaxtyping import Float, UInt
from matplotlib.artist import Artist
from mpl_toolkits.mplot3d import Axes3D

from cleo.base import Recorder, SynapseDevice
from cleo.coords import coords_from_ng
from cleo.utilities import (
    analog_signal,
    normalize_coords,
    rng,
    unit_safe_append,
    unit_safe_cat,
)


def target_neurons_in_plane(
    ng: NeuronGroup,
    scope_focus_depth: Quantity,
    scope_img_width: Quantity,
    scope_location: Quantity = (0, 0, 0) * mm,
    scope_direction: tuple = (0, 0, 1),
    soma_radius: Quantity = 10 * um,
    sensor_location: str = "cytoplasm",
) -> tuple[UInt[np.ndarray, "n_targets"], Float[np.ndarray, "n_targets"], Quantity]:
    """
    Returns a tuple of (i_targets, noise_focus_factor, coords_on_plane)

    Parameters
    ----------
    ng : NeuronGroup
        The neuron group to target.
    scope_focus_depth : Quantity
        The depth of the focal plane of the microscope.
    scope_img_width : Quantity
        The width of the image captured by the microscope.
    scope_location : Quantity, optional
        The location of the microscope, by default (0, 0, 0) * mm.
    scope_direction : tuple, optional
        The direction of the microscope, by default (0, 0, 1).
    soma_radius : Quantity, optional
        The radius of the soma of the neuron, by default 10 * um.
        Used to compute noise focus factor, since smaller ROIs will have
        a noisier distribution of fluorescence, averaged over fewer pixels.
    sensor_location : str, optional
        The location of the sensor, by default "cytoplasm".

    Returns
    -------
    Tuple[UInt[np.ndarray, "n_targets"], Float[np.ndarray, "n_targets"], Quantity]
        A tuple of (i_targets, noise_focus_factor, coords_on_plane)
    """
    assert sensor_location in ("cytoplasm", "membrane")

    ng_coords = coords_from_ng(ng)
    # Compute the normal vector and the center of the plane
    plane_normal = scope_direction
    plane_center = scope_location + plane_normal * scope_focus_depth
    # Compute the distance of each neuron from the plane
    perp_distances = np.abs(np.dot(ng_coords - plane_center, plane_normal))
    noise_focus_factor = np.ones(ng.N)
    # signal falloff with shrinking cross-section (or circumference)
    # ignore numpy warning
    with np.errstate(invalid="ignore"):
        r_soma_visible = np.sqrt(soma_radius**2 - perp_distances**2)
    r_soma_visible[np.isnan(r_soma_visible)] = 0
    if sensor_location == "cytoplasm":
        relative_num_pixels = (r_soma_visible / soma_radius) ** 2
    else:
        relative_num_pixels = r_soma_visible / soma_radius
    with np.errstate(divide="ignore"):
        noise_focus_factor /= np.sqrt(relative_num_pixels)

    # get only neurons in view
    coords_on_plane = ng_coords - plane_normal * perp_distances[:, np.newaxis]
    plane_distances = np.sqrt(np.sum((coords_on_plane - plane_center) ** 2, axis=1))

    i_targets = np.flatnonzero(
        np.logical_and(r_soma_visible > 0, plane_distances < scope_img_width / 2)
    )

    return i_targets, noise_focus_factor[i_targets], coords_on_plane[i_targets]


@define(eq=False)
class Scope(Recorder):
    """Two-photon microscope.

    Injection kwargs
    ----------------
    rho_rel_generator : Callable[[int], Float[np.ndarray, "n"], optional
        A function assigning expression levels. Takes n as an arg, outputs float array.
        ``lambda n: np.ones(n)`` by default.
    focus_depth : Quantity, optional
        The depth of the focal plane, by default that of the scope.
    soma_radius : Quantity, optional
        The radius of the soma of the neuron, by default that of the scope.
        Used to compute noise focus factor, since smaller ROIs will have
        a noisier distribution of fluorescence, averaged over fewer pixels.
    """

    sensor: SynapseDevice = field()
    img_width: Quantity = field()
    """The width (diameter) of the (circular) image captured by the microscope.
    Specified in distance units."""
    focus_depth: Quantity = None
    """The depth of the focal plane, with distance units"""
    location: Quantity = [0, 0, 0] * mm
    """Location of the objective lens."""
    direction: Float[np.ndarray, "3"] = field(
        default=(0, 0, 1), converter=normalize_coords
    )
    """Direction in which the microscope is pointing.
    By default straight down (`+z` direction)"""
    soma_radius: Quantity = field(default=10 * um)
    """Assumed radius of neurons, used to compute noise focus factor.
    Smaller neurons have noisier signals."""
    snr_cutoff: float = field(default=1)
    """SNR below which neurons are discarded.
    Applied only when not focus_depth is not None"""
    dFF: list[Float[np.ndarray, "{self.n}"]] = field(
        factory=list, init=False, repr=False
    )
    """ΔF/F from every call to :meth:`get_state`.
    Shape is (n_samples, n_ROIs). Stored if :attr:`~cleo.InterfaceDevice.save_history`"""
    t: Quantity = field(factory=lambda: ms * [], init=False, repr=False)
    """Times at which sensor traces are recorded, with Brian units, stored if
    :attr:`~cleo.InterfaceDevice.save_history`"""

    neuron_groups: list[NeuronGroup] = field(factory=list, repr=False, init=False)
    """neuron groups the scope has been injected into, in order of injection"""
    i_targets_per_injct: list[UInt[np.ndarray, "*"]] = field(
        factory=list, repr=False, init=False
    )
    """targets of neurons selected from each injection"""
    sigma_per_injct: list[Float[np.ndarray, "*"]] = field(
        factory=list, repr=False, init=False
    )
    """`sigma_noise` of neurons selected from each injection"""
    focus_coords_per_injct: list[Float[np.ndarray, "*"]] = field(
        factory=list, repr=False, init=False
    )
    """coordinates on the focal plane of neurons selected from each injection"""
    rho_rel_per_injct: list[Float[np.ndarray, "*"]] = field(
        factory=list, repr=False, init=False
    )
    """relative expression levels of neurons selected from each injection"""

    @property
    def n(self) -> int:
        """Number of imaged ROIs"""
        return sum(len(i_t) for i_t in self.i_targets_per_injct)

    @property
    def sigma_noise(self) -> Float[np.ndarray, "{self.n}"]:
        """noise std dev (in terms of ΔF/F) for all targets, in order injected."""
        if self.n == 0:
            return np.array([])
        return np.concatenate(self.sigma_per_injct)

    @property
    def focus_coords(self) -> Quantity:
        """coordinates on the focal plane of all targets, in order injected."""
        if self.n == 0:
            return [] * mm
        return unit_safe_cat(self.focus_coords_per_injct)

    @property
    def dFF_1AP(self) -> Float[np.ndarray, "{self.n}"]:
        """dFF_1AP for all targets, in order injected. Varies with expression levels."""
        if self.n == 0:
            return np.array([])
        rho_rel = np.concatenate(self.rho_rel_per_injct)
        assert rho_rel.shape == (self.n,)
        return rho_rel * self.sensor.dFF_1AP

    def _init_saved_vars(self):
        if self.save_history:
            self.t = fields(type(self)).t.default.factory()
            self.dFF = fields(type(self)).dFF.default.factory()

    def _update_saved_vars(self, t, dFF):
        if self.save_history:
            self.t = unit_safe_append(self.t, t)
            self.dFF.append(dFF)

    def __attrs_post_init__(self):
        self._init_saved_vars()

    def reset(self, **kwargs) -> None:
        self._init_saved_vars()

    def target_neurons_in_plane(
        self, ng, focus_depth: Quantity = None, soma_radius: Quantity = None
    ) -> tuple[
        UInt[np.ndarray, "n_targets"],
        Float[np.ndarray, "n_targets"],
        Float[np.ndarray, "n_targets"],
    ]:
        """calls :func:`target_neurons_in_plane` with scope parameter defaults.
        `focus_depth` and `soma_radius` can be overridden here.

        Parameters
        ----------
        ng : NeuronGroup
            The neuron group to target.
        focus_depth : Quantity
            The depth of the focal plane, by default that of the microscope.
        soma_radius : Quantity, optional
            The radius of the soma of the neuron, by default that of the microscope.
            Used to compute noise focus factor, since smaller ROIs will have
            a noisier distribution of fluorescence, averaged over fewer pixels.

        Returns
        -------
        Tuple[UInt[np.ndarray, "n_targets"], Float[np.ndarray, "n_targets"], Float[np.ndarray, "n_targets"]]
            A tuple of (i_targets, noise_focus_factor, coords_on_plane)
        """
        focus_depth = focus_depth or self.focus_depth
        soma_radius = soma_radius or self.soma_radius
        return target_neurons_in_plane(
            ng,
            focus_depth,
            self.img_width,
            self.location,
            self.direction,
            soma_radius,
            self.sensor.location,
        )

    def get_state(self) -> Float[np.ndarray, "{self.n}"]:
        """Returns a 1D array of ΔF/F values for all targets, in order injected.

        Returns
        -------
        Float[np.ndarray, "{self.n}"]
            Fluorescence values for all targets
        """
        signal = []
        signal_per_ng = self.sensor.get_state()
        # sensor has just one signal for neuron group, not storing
        # separately for each injection, so we'll recover that here
        n_prev_targets_for_ng = {}
        for ng, i_targets in zip(self.neuron_groups, self.i_targets_per_injct):
            if ng.name not in signal_per_ng:
                raise RuntimeError(
                    f"Sensor {self.sensor.name} has no signal for neuron group {ng.name}."
                    " Did you forget to call inject_sensor_for_targets() after scope injections??"
                )
            subset_start = n_prev_targets_for_ng.get(ng, 0)
            subset_for_injct = slice(subset_start, subset_start + len(i_targets))
            signal.append(signal_per_ng[ng.name][subset_for_injct])
            n_prev_targets_for_ng[ng] = subset_start + len(i_targets)

        signal = np.concatenate(signal)
        noise = rng.normal(0, self.sigma_noise, len(signal))
        assert self.n == len(signal) == len(noise) == len(self.sigma_noise)

        state = signal + noise
        t_now = self.sim.network.t
        self._update_saved_vars(t_now, state)
        return signal + noise

    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams) -> None:
        focus_depth = kwparams.get("focus_depth", self.focus_depth)
        soma_radius = kwparams.get("soma_radius", self.soma_radius)
        rho_rel_generator = kwparams.get("rho_rel_generator", lambda n: np.ones(n))
        if focus_depth:
            if "i_targets" in kwparams or "sigma_noise" in kwparams:
                raise ValueError(
                    "i_targets and sigma_noise can not be specified while focus_depth",
                    " is not None",
                )
            (
                i_targets,
                noise_focus_factor,
                focus_coords,
            ) = self.target_neurons_in_plane(neuron_group, focus_depth, soma_radius)
            base_sigma = kwparams.get("base_sigma", self.sensor.sigma_noise)
            sigma_noise = noise_focus_factor * base_sigma
            rho_rel = rho_rel_generator(len(i_targets))
            if self.sensor.dFF_1AP is not None:
                snr = rho_rel * self.sensor.dFF_1AP / sigma_noise
                i_targets = i_targets[snr > self.snr_cutoff]
                sigma_noise = sigma_noise[snr > self.snr_cutoff]
                focus_coords = focus_coords[snr > self.snr_cutoff]
                rho_rel = rho_rel[snr > self.snr_cutoff]
            else:
                warnings.warn(
                    f"SNR cutoff not used, since {self.sensor.name} does not have dFF_1AP defined."
                )
        else:
            i_targets = kwparams.pop("i_targets", neuron_group.i_)
            sigma_noise = kwparams.get("sigma_noise", self.sensor.sigma_noise)
            rho_rel = rho_rel_generator(len(i_targets))
            _, sigma_noise, rho_rel = np.broadcast_arrays(
                i_targets, sigma_noise, rho_rel
            )
            focus_coords = coords_from_ng(neuron_group)[i_targets]
        assert len(i_targets) == len(sigma_noise) == len(focus_coords) == len(rho_rel)

        self.neuron_groups.append(neuron_group)
        self.i_targets_per_injct.append(i_targets)
        self.sigma_per_injct.append(sigma_noise)
        self.focus_coords_per_injct.append(focus_coords)
        self.rho_rel_per_injct.append(rho_rel)

    def i_targets_for_neuron_group(self, neuron_group):
        """can handle multiple injections into same ng"""
        i_targets_for_ng = []
        for ng, i_targets in zip(self.neuron_groups, self.i_targets_per_injct):
            if ng is neuron_group:
                i_targets_for_ng.extend(i_targets)
        return i_targets_for_ng

    def inject_sensor_for_targets(self, **kwparams) -> None:
        for ng in set(self.neuron_groups):
            i_targets_for_ng = []
            rho_rel_for_ng = []
            for ng_, i_targets, rho_rel in zip(
                self.neuron_groups, self.i_targets_per_injct, self.rho_rel_per_injct
            ):
                if ng_ is not ng:
                    continue

                i_targets_for_ng.extend(i_targets)
                rho_rel_for_ng.extend(rho_rel)

            self.sim.inject(
                self.sensor,
                ng,
                i_targets=i_targets_for_ng,
                rho_rel=rho_rel_for_ng,
                **kwparams,
            )

    def add_self_to_plot(
        self, ax: Axes3D, axis_scale_unit: Unit, **kwargs
    ) -> list[Artist]:
        color = kwargs.pop("color", "xkcd:fluorescent green")
        snr = np.concatenate(self.sigma_per_injct)
        coords = (
            np.concatenate([
                coords_from_ng(ng)[i_targets] / meter
                for ng, i_targets in zip(self.neuron_groups, self.i_targets_per_injct)
            ])
            * meter
        )
        assert coords.shape == (len(snr), 3)
        assert len(snr) == len(coords)
        scope_marker = ax.quiver(
            self.location[0] / axis_scale_unit,
            self.location[1] / axis_scale_unit,
            self.location[2] / axis_scale_unit,
            self.direction[0],
            self.direction[1],
            self.direction[2],
            color=color,
            lw=5,
            label=self.name,
            pivot="tail",
            length=self.focus_depth / axis_scale_unit,
            normalize=True,
        )

        # Define the center, normal vector, and radius
        normal = self.direction
        center = self.location + normal * self.focus_depth
        radius = self.img_width / 2

        # Generate the points for the circle
        theta = np.linspace(0, 2 * np.pi, 100)
        r = np.sqrt(np.linspace(0, 1, 5))
        # angle away from z axis
        phi = np.arccos(normal[2])
        # angle relative to x axis
        theta0 = np.arctan2(normal[1], normal[0])
        x = center[0] + radius * np.cos(phi) * np.outer(np.cos(theta), r)
        y = center[1] + radius * np.cos(phi) * np.outer(np.sin(theta), r)
        z = center[2] + radius * np.sin(phi) * np.outer(-np.cos(theta - theta0), r)

        plane = ax.plot_surface(
            x / axis_scale_unit,
            y / axis_scale_unit,
            z / axis_scale_unit,
            color=color,
            alpha=0.2,
        )

        target_markers = ax.scatter(
            coords[:, 0] / axis_scale_unit,
            coords[:, 1] / axis_scale_unit,
            coords[:, 2] / axis_scale_unit,
            marker="^",
            c=color,
            label=f"{self.sensor.name} ROIs",
            **kwargs,
        )
        color_rgba = target_markers.get_facecolor()
        color_rgba[:, :3] = 0.3 * color_rgba[:, :3]
        target_markers.set(color=color_rgba)

        handles = ax.get_legend().legend_handles
        handles.append(target_markers)
        patch = mpl.patches.Patch(color=color, label=self.name)
        handles.append(patch)
        ax.legend(handles=handles)

        return [scope_marker, target_markers, plane]

    def to_neo(self) -> neo.AnalogSignal:
        # inherit docstring
        try:
            signal = analog_signal(self.t, self.dFF, "")
        except AttributeError:
            return
        signal.name = self.name
        signal.description = (
            "Fluorescence trace for each ROI. Exported from Cleo Scope device"
        )
        signal.annotate(
            export_datetime=datetime.now(),
            sensor=self.sensor.name,
            scope_direction=self.direction,
            scope_location=self.location / um * pq.um,
            focus_depth=self.focus_depth / um * pq.um,
            img_width=self.img_width / um * pq.um,
        )
        if self.n > 0:
            focus_coords = self.focus_coords / um * pq.um
            signal.array_annotate(
                dFF_1AP=self.dFF_1AP,
                x=focus_coords[..., 0],
                y=focus_coords[..., 1],
                z=focus_coords[..., 2],
                i_roi=np.arange(self.n),
            )
        return signal
        # want ROI coordinates and size in image
