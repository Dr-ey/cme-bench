"""Explainers. Importing this package fills methods.base.REGISTRY."""
from cmebench.methods import legacy  # noqa: F401  (CME, CAM, Grad-CAM family, Guided, Random)
from cmebench.methods import extremal_perturbation, libragrad, meaningful_perturbation, rise, scorecam  # noqa: F401
