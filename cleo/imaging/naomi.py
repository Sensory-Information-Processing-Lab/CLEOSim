from __future__ import annotations
from attrs import define, field, fields_dict, asdict
from brian2 import Synapses, np, Quantity, NeuronGroup, second, umolar, nmolar
from cleo.base import SynapseDevice
from cleo.utilities import brian_safe_name


@define(eq=False)
class NAOMiSensor(SynapseDevice):
    """Base class for NAOMi sensors"""

    sigma_noise: float = field(kw_only=True)
    dFF_1AP: float = field(kw_only=True)
    location: str = field(kw_only=True)

    @location.validator
    def _check_location(self, attribute, value):
        if value not in ("cytoplasm", "membrane"):
            raise ValueError(
                f"Indicator location must be 'cytoplasm' or 'membrane', not {value}"
            )

    @property
    def snr(self) -> float:
        """Signal-to-noise ratio for 1 AP"""
        return self.dFF_1AP / self.sigma_noise

    def get_state(self) -> dict[NeuronGroup, np.ndarray]:
        """Returns a {neuron_group: fluorescence} dict of dFF values."""
        pass

    @property
    def exc_spectrum(self) -> list[tuple[float, float]]:
        """Excitation spectrum, alias for :attr:`spectrum`"""
        return self.spectrum


@define(eq=False)
class NAOMiCalciumModel:
    """Base class for how NAOMi GECI computes calcium concentration."""

    on_pre: str = field(default="", init=False)
    model: str


@define(eq=False)
class NAOMiPreexistingCalcium(NAOMiCalciumModel):
    """Calcium concentration is pre-existing in neuron model."""

    model: str = field(default="Ca = Ca_pre : mmolar", init=False)


@define(eq=False)
class NAOMiDynamicCalcium(NAOMiCalciumModel):
    """Simulates intracellular calcium dynamics from spikes."""

    on_pre: str = field(default="Ca += dCa_T / (1 + kappa_S + kappa_B)", init=False)
    model: str = field(
        default="""
            dCa/dt = -gamma * (Ca - Ca_rest) / (1 + kappa_S + kappa_B) : mmolar (clock-driven)
            kappa_B = B_T * K_d / (Ca + K_d)**2 : 1""",
        init=False,
    )

    Ca_rest: Quantity = field(kw_only=True)
    gamma: Quantity = field(kw_only=True)
    B_T: Quantity = field(kw_only=True)
    kappa_S: float = field(kw_only=True)
    dCa_T: Quantity = field(kw_only=True)

    def init_syn_vars(self, syn: Synapses) -> None:
        syn.Ca = self.Ca_rest


@define(eq=False)
class NAOMiCalBindingActivationModel:
    """Base class for modeling calcium binding/activation in NAOMi."""

    model: str


@define(eq=False)
class NAOMiNullBindingActivation(NAOMiCalBindingActivationModel):
    """Doesn't model binding/activation; goes straight from [Ca2+] to ΔF/F."""

    model: str = field(default="CaB_active = Ca: mmolar", init=False)


@define(eq=False)
class NAOMiDoubExpCalBindingActivation(NAOMiCalBindingActivationModel):
    """Double exponential kernel convolution representing CaB binding/activation."""

    model: str = field(
        default="""
            CaB_active = Ca_rest + b : mmolar  # add tiny bit to avoid /0
            db/dt = beta : mmolar (clock-driven)
            lam = 1/tau_off + 1/tau_on : 1/second
            kap = 1/tau_off : 1/second
            dbeta/dt = (
                A * (lam - kap) * (Ca - Ca_rest)
                - (kap + lam) * beta
                - kap * lam * b
            ) : mmolar/second (clock-driven)
            """,
        init=False,
    )

    A: float = field(kw_only=True)
    tau_on: Quantity = field(kw_only=True)
    tau_off: Quantity = field(kw_only=True)
    Ca_rest: Quantity = field(kw_only=True)

    def init_syn_vars(self, syn: Synapses) -> None:
        syn.b = 0
        syn.beta = 0


@define(eq=False)
class NAOMiExcitationModel:
    """Defines exc_factor in NAOMi."""

    model: str


@define(eq=False)
class NAOMiNullExcitation(NAOMiExcitationModel):
    """Models excitation as a constant factor in NAOMi."""

    model: str = field(default="exc_factor = 1 : 1", init=False)


@define(eq=False)
class NAOMiGECI(NAOMiSensor):
    """NAOMi GECI model based on Song et al., 2021, with interchangeable components."""

    location: str = field(default="cytoplasm", init=False)
    cal_model: NAOMiCalciumModel = field(kw_only=True)
    bind_act_model: NAOMiCalBindingActivationModel = field(kw_only=True)
    exc_model: NAOMiExcitationModel = field(kw_only=True)

    fluor_model: str = field(
        default="""
            dFF_baseline = 1 / (1 + (K_d / Ca_rest) ** n_H) : 1
            dFF = exc_factor * rho_rel * dFF_max  * (
                1 / (1 + (K_d / CaB_active) ** n_H)
                - dFF_baseline
            ) : 1
            rho_rel : 1
        """,
        init=False,
    )

    K_d: Quantity = field(kw_only=True)
    n_H: float = field(kw_only=True)
    dFF_max: float = field(kw_only=True)

    def get_state(self) -> dict[NeuronGroup, np.ndarray]:
        return {ng_name: syn.dFF for ng_name, syn in self.synapses.items()}

    def __attrs_post_init__(self):
        self.model = "\n".join([
            self.cal_model.model,
            self.bind_act_model.model,
            self.exc_model.model,
            self.fluor_model,
        ])
        self.on_pre = self.cal_model.on_pre

    def init_syn_vars(self, syn: Synapses) -> None:
        for model in [self.cal_model, self.bind_act_model, self.exc_model]:
            if hasattr(model, "init_syn_vars"):
                model.init_syn_vars(syn)

    @property
    def params(self) -> dict:
        """Returns a dictionary of all parameters from model/submodels"""
        params = asdict(self, recurse=False)
        for field in fields_dict(NAOMiSensor):
            params.pop(field)
        for key in list(params.keys()):
            if key.startswith("_"):
                params.pop(key)
        for model in [self.cal_model, self.bind_act_model, self.exc_model]:
            to_add = asdict(model, recurse=False)
            to_add.pop("model")
            params.update(to_add)
        return params


def geci_naomi(doub_exp_conv: bool, pre_existing_cal: bool, **kwparams) -> NAOMiGECI:
    ExcModel = NAOMiNullExcitation
    CalModel = NAOMiPreexistingCalcium if pre_existing_cal else NAOMiDynamicCalcium
    BAModel = (
        NAOMiDoubExpCalBindingActivation
        if doub_exp_conv
        else NAOMiNullBindingActivation
    )

    def init_from_kwparams(cls, **more_kwargs):
        kwparams.update(more_kwargs)
        kwparams_to_keep = {}
        for field_name, field in fields_dict(cls).items():
            if field.init and field_name in kwparams:
                kwparams_to_keep[field_name] = kwparams[field_name]
        return cls(**kwparams_to_keep)

    NAOMiGECIClass = NAOMiGECI

    return init_from_kwparams(
        NAOMiGECIClass,
        cal_model=init_from_kwparams(CalModel),
        exc_model=init_from_kwparams(ExcModel),
        bind_act_model=init_from_kwparams(BAModel),
        **kwparams,
    )


def _create_naomi_geci_fn(
    name,
    K_d,
    n_H,
    dFF_max,
    sigma_noise_rel,
    dFF_1AP_rel=None,
    ca_amp=None,
    t_on=None,
    t_off=None,
    extra_doc="",
):
    gcamp6s_dFF_1AP_dana2019 = 0.133
    gcamp6s_snr_1AP_dana2019 = 4.4
    sigma_gcamp6s = gcamp6s_dFF_1AP_dana2019 / gcamp6s_snr_1AP_dana2019
    sigma_noise = sigma_noise_rel * sigma_gcamp6s
    if dFF_1AP_rel is not None:
        dFF_1AP = dFF_1AP_rel * gcamp6s_dFF_1AP_dana2019
    else:
        dFF_1AP = None

    def naomi_geci_fn(
        doub_exp_conv=True,
        pre_existing_cal=False,
        K_d=K_d * nmolar,
        n_H=n_H,
        dFF_max=dFF_max,
        sigma_noise=sigma_noise,
        dFF_1AP=dFF_1AP,
        ca_amp=ca_amp,
        t_on=t_on,
        t_off=t_off,
        Ca_rest=50 * nmolar,
        kappa_S=110,
        gamma=292.3 / second,
        B_T=200 * umolar,
        dCa_T=7.6 * umolar,
        name=name,
        **kwparams,
    ) -> NAOMiGECI:
        A = ca_amp / (second / 100) if ca_amp else None
        tau_on = second / t_on if t_on else None
        tau_off = second / t_off if t_off else None

        return geci_naomi(
            doub_exp_conv,
            pre_existing_cal,
            K_d=K_d,
            n_H=n_H,
            dFF_max=dFF_max,
            sigma_noise=sigma_noise,
            dFF_1AP=dFF_1AP,
            A=A,
            tau_on=tau_on,
            tau_off=tau_off,
            Ca_rest=Ca_rest,
            kappa_S=kappa_S,
            gamma=gamma,
            B_T=B_T,
            dCa_T=dCa_T,
            name=name,
            **kwparams,
        )

    if naomi_geci_fn.__doc__ is None:
        naomi_geci_fn.__doc__ = ""

    naomi_geci_fn.__doc__ += "\n\n" + (" " * 8) + extra_doc

    globals()[brian_safe_name(name.lower())] = naomi_geci_fn


_create_naomi_geci_fn(
    "GCaMP3_NAOMi", 287, 2.52, 12, (3.9 / 2.1) / (13.3 / 4.4), 3.9 / 13.3, 0.05, 1, 1
)
_create_naomi_geci_fn(
    "GCaMP6-RS09_NAOMi",
    520,
    3.2,
    25,
    1,
    extra_doc="",
)
_create_naomi_geci_fn(
    "GCaMP6-RS06_NAOMi",
    320,
    3,
    15,
    1,
    extra_doc="",
)
_create_naomi_geci_fn(
    "GCaMP6f_NAOMi", 290, 2.7, 25.2, 1.24, 0.735, 76.1251, 0.8535, 98.6173
)
_create_naomi_geci_fn("GCaMP6s_NAOMi", 147, 2.45, 27.2, 1, 1, 54.6943, 0.4526, 68.5461)
_create_naomi_geci_fn("jGCaMP7b_NAOMi", 82, 3.06, 22.1, 0.25, 4.64)
_create_naomi_geci_fn("jGCaMP7c_NAOMi", 298, 2.44, 145.6, 0.39, 1.85)
_create_naomi_geci_fn("jGCaMP7f_NAOMi", 174, 2.3, 30.2, 0.72, 1.71)
_create_naomi_geci_fn("jGCaMP7s_NAOMi", 68, 2.49, 40.4, 0.33, 4.96)
_create_naomi_geci_fn(
    "OGB-1_naomi",
    250,
    1,
    14,
    1,
    extra_doc="",
)
