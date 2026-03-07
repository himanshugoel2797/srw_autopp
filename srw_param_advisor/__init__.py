"""
SRW Parameter Advisor
=====================

AI-driven propagation parameter optimization for the Synchrotron Radiation
Workshop (SRW) wavefront propagation code.

Two-stage approach:
  1. Analytical estimator: ABCD matrix trace through optical layout
  2. RL-trained ViT agent: refines parameters based on actual wavefront structure

Components:
  - validator: Post-propagation quality checking with actionable diagnostics
  - analytical: Rule-based parameter estimation from beam optics
  - agent: RL contextual bandit with mode-conditional policy
  - preprocessing: Physics-normalised spatial map preparation
  - wavefront_gen: Universal parametric source for training data
"""

__version__ = "0.1.0"
