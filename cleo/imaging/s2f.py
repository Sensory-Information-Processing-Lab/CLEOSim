import sympy
import warnings

from brian2 import np, NeuronGroup, ms, second, nmeter, Quantity
from brian2.monitors import SpikeMonitor
from attrs import define, field, fields_dict

from cleo.base import SynapseDevice
from cleo.light import LightDependent
from cleo.utilities import brian_safe_name
from cleo.coords import coords_from_ng


@define(eq=False, slots=False)
class S2FGECI(SynapseDevice):
    """Base class for S2F GECI models"""

    rise: Quantity = field(kw_only=True)
    decay1: Quantity = field(kw_only=True)
    decay2: Quantity = field(kw_only=True, default=0 * ms)
    r: float = field(kw_only=True, default=0)
    Fm: float = field(kw_only=True, default=1.0)
    Ca0: float = field(kw_only=True, default=0.0)
    beta: float = field(kw_only=True, default=1.0)
    F0: float = field(kw_only=True, default=0.0)
    sigma_noise: float = field(kw_only=True, default=0.0)
    threshold: float = field(kw_only=True, default=1e-3)
    spike_monitors: dict[NeuronGroup, SpikeMonitor] = field(factory=dict, init=False)
    i_targets: dict[str, np.ndarray] = field(factory=dict, init=False)
    dFF_1AP: float = field(default=None, kw_only=True)
    location: str = field(default="cytoplasm", init=False)
    ignore_time: Quantity = field(init=False)

    def _find_ignore_time(self) -> Quantity:
        """Calculates time before old spikes become insignificant to save computation."""
        t_k = sympy.Symbol("t_k")

        expr = (
            sympy.exp(t_k / (self.decay1 / second))
            + self.r * sympy.exp(t_k / (self.decay2 / second))
        ) * (1 - sympy.exp(t_k / (self.rise / second))) - self.threshold

        try:
            sol = sympy.nsolve(expr, t_k, -2)
        except RecursionError as e:
            warnings.warn(
                f"S2FGECI._find_ignore_time failed: {e}. This likely means that rise, decay1, or decay2 units were not set. Defaulting to 15 ms for spike ignore time."
            )
            sol = -15e-3
        except ValueError as e:
            warnings.warn(
                f"S2FGECI._find_ignore_time failed: {e}. Check the values of rise, decay1, decay2, and r. Defaulting to 15 ms for spike ignore time."
            )
            sol = -15e-3

        return -float(sol) * second

    def __attrs_post_init__(self):
        self.ignore_time = self._find_ignore_time()

    @location.validator
    def _check_location(self, attribute, value):
        if value not in ("cytoplasm", "membrane"):
            raise ValueError(
                f"Indicator location must be 'cytoplasm' or 'membrane', not {value}"
            )

    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwargs):
        """
        Connects a neuron group to the S2F model by setting up a SpikeMonitor.

        Parameters:
        - neuron_group: The NeuronGroup to monitor for spikes.
        """
        # Create a SpikeMonitor for the neuron group
        spike_monitor = SpikeMonitor(neuron_group)

        # Store the SpikeMonitor in a dictionary keyed by the neuron group's name
        self.spike_monitors[neuron_group.name] = spike_monitor
        self.source_ngs[neuron_group.name] = neuron_group
        if neuron_group.name in self.i_targets:
            self.i_targets[neuron_group.name] = np.append(
                self.i_targets[neuron_group.name],
                kwargs.get("i_targets", []),
            )
        else:
            self.i_targets[neuron_group.name] = kwargs.get(
                "i_targets", np.arange(neuron_group.N)
            )

        # Add the SpikeMonitor to the simulation's objects if necessary
        # This ensures the monitor is updated during the simulation
        if hasattr(self, "brian_objects"):
            self.brian_objects.add(spike_monitor)

    def spike_to_calcium(self, spike_times):
        """Converts spike times to calcium response using the S2F model."""
        calcium_response = np.zeros_like(spike_times)
        t_diffs = self.sim.network.t - spike_times
        ignored_indices = t_diffs <= self.ignore_time
        calcium_response[ignored_indices] = (
            np.exp(-t_diffs[ignored_indices] / self.decay1)
            + self.r * np.exp(-t_diffs[ignored_indices] / self.decay2)
        ) * (1 - np.exp(-t_diffs[ignored_indices] / self.rise))

        calcium_response += np.random.normal(
            0, self.sigma_noise, size=calcium_response.shape
        )
        return np.maximum(calcium_response, 0)

    def sigmoid_response(self, ca_trace):
        """Applies a sigmoidal transformation to the calcium trace."""
        return self.Fm / (1 + np.exp(-self.beta * (ca_trace - self.Ca0))) + self.F0

    def get_response(self, ng_name):
        """Computes calcium traces and sum over time"""
        target_spike_indices = np.isin(
            self.spike_monitors[ng_name].i, self.i_targets[ng_name]
        )
        calcium_responses = self.spike_to_calcium(
            self.spike_monitors[ng_name].t[target_spike_indices]
        )

        neuron_indices = self.spike_monitors[ng_name].i[target_spike_indices]
        J = np.eye(self.source_ngs[ng_name].N)[neuron_indices]

        ca_trace = (calcium_responses @ J)[self.i_targets[ng_name]]
        return self.sigmoid_response(ca_trace)

    def get_state(self) -> dict[NeuronGroup, np.ndarray]:
        return {ng_name: self.get_response(ng_name) for ng_name in self.spike_monitors}


@define(eq=False, slots=False)
class S2FLightDependentGECI(S2FGECI, LightDependent):
    """S2F GECI model with light-dependent adjustments"""

    spectrum: list[tuple[float, float]] = field(kw_only=True)

    def get_response(self, ng_name):
        light_intensity = self.light_agg_ngs.transmittance(
            coords_from_ng(self.source_ngs[ng_name])
        )
        epsilon = self.epsilon(self.light_source.wavelength / nmeter)
        adjusted_light_intensity = light_intensity * epsilon
        return super().get_response(ng_name) * adjusted_light_intensity


def geci_s2f(name: str, light_dependent: bool = False, **kwparams) -> S2FGECI:
    """Initializes an S2F GECI model with given parameters.

    Parameters
    ----------
    name : str
        Name of the GECI model
    light_dependent : bool
        Whether the indicator is light-dependent

    Returns
    -------
    S2FGECI
        A S2F(LightDependent)GECI model with specified parameters
    """
    GECIClass = S2FLightDependentGECI if light_dependent else S2FGECI

    # Filter kwparams to only include fields relevant to the chosen class
    kwparams_to_keep = {}
    for field_name, field in fields_dict(GECIClass).items():
        if field.init and field_name in kwparams:
            kwparams_to_keep[field_name] = kwparams[field_name]

    return GECIClass(name=name, **kwparams_to_keep)


def _create_s2f_geci_fn(
    name,
    rise,
    decay1,
    decay2,
    r,
    Fm,
    Ca0,
    beta,
    F0,
    sigma_noise,
):
    def s2f_geci_fn(light_dependent=False, light_source=None, spectrum=[]):
        if light_dependent:
            return S2FLightDependentGECI(
                name=name,
                rise=rise,
                decay1=decay1,
                decay2=decay2,
                r=r,
                Fm=Fm,
                Ca0=Ca0,
                beta=beta,
                F0=F0,
                sigma_noise=sigma_noise,
                spectrum=spectrum,
                light_source=light_source,
            )
        else:
            return S2FGECI(
                name=name,
                rise=rise,
                decay1=decay1,
                decay2=decay2,
                r=r,
                Fm=Fm,
                Ca0=Ca0,
                beta=beta,
                F0=F0,
                sigma_noise=sigma_noise,
            )

    globals()[brian_safe_name(name.lower())] = s2f_geci_fn


# Define specific S2F GECI functions based on the chart
_create_s2f_geci_fn(
    "jGCaMP8f_S2F",
    1.85 * ms,
    34.07 * ms,
    263.70 * ms,
    0.48,
    6.104380,
    4.170575,
    0.390533,
    -1.001000,
    0.03,
)
_create_s2f_geci_fn(
    "jGCaMP8m_S2F",
    2.46 * ms,
    41.64 * ms,
    245.80 * ms,
    0.28,
    7.454645,
    2.691117,
    0.360008,
    -2.050880,
    0.03,
)
_create_s2f_geci_fn(
    "jGCaMP8s_S2F",
    5.65 * ms,
    86.26 * ms,
    465.45 * ms,
    0.19,
    7.455792,
    1.282417,
    0.343721,
    -2.919320,
    0.03,
)
_create_s2f_geci_fn(
    "jGCaMP7f_S2F",
    16.21 * ms,
    95.27 * ms,
    398.22 * ms,
    0.24,
    6.841247,
    5.562159,
    0.423212,
    -0.593480,
    0.03,
)
_create_s2f_geci_fn(
    "XCaMP-Gf_S2F",
    13.93 * ms,
    99.38 * ms,
    312.85 * ms,
    0.20,
    2.363793,
    3.936075,
    0.471668,
    -0.319370,
    0.03,
)
_create_s2f_geci_fn(
    "GCaMP6s_S2F",
    50.81 * ms,
    1702.21 * ms,
    0.00 * ms,
    0.00,
    3.334000,
    3.142000,
    1.332000,
    -0.049982,
    0.03,
)
_create_s2f_geci_fn(
    "GCaMP6s-TG_S2F",
    133.01 * ms,
    1262.78 * ms,
    0.00 * ms,
    0.00,
    3.596000,
    3.303000,
    2.897000,
    -0.000251,
    0.03,
)
_create_s2f_geci_fn(
    "GCaMP6f_S2F",
    9.98 * ms,
    682.58 * ms,
    0.00 * ms,
    0.00,
    1.905000,
    3.197000,
    1.410000,
    -0.020769,
    0.03,
)
_create_s2f_geci_fn(
    "GCaMP6f-TG_S2F",
    20.82 * ms,
    629.74 * ms,
    0.00 * ms,
    0.00,
    2.818000,
    5.821000,
    1.046000,
    -0.006377,
    0.03,
)
