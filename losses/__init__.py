"""
losses package

Re-exports the loss/cost utilities from losses.py so that
matcher.py and criterion.py can do `from losses import (...)`.

NOTE: HungarianMatcher and SetCriterion are intentionally NOT
re-exported here (import them directly via
`from losses.matcher import HungarianMatcher` /
`from losses.criterion import SetCriterion`) to avoid a circular
import: matcher.py and criterion.py both import from this same
`losses` package, so pulling them in here too would create a
package <-> submodule import cycle.
"""

from .losses import (
    dice_loss,
    sigmoid_ce_loss,
    batch_dice_cost,
    batch_sigmoid_ce_cost,
    calculate_uncertainty,
    sample_points,
    sample_uncertain_points,
)

__all__ = [
    "dice_loss",
    "sigmoid_ce_loss",
    "batch_dice_cost",
    "batch_sigmoid_ce_cost",
    "calculate_uncertainty",
    "sample_points",
    "sample_uncertain_points",
]
