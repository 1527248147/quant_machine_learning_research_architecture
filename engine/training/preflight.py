"""
Preflight validation: check model-label contract before training.

This catches configuration errors early — before any data is loaded or
any GPU memory is allocated.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from engine.core.exceptions import ContractViolation
from engine.schema.contracts import ModelLabelContract
from engine.targets.specs import TargetSpec

logger = logging.getLogger(__name__)


def validate_contract(
    model_name: str,
    contract: ModelLabelContract,
    label_names: List[str],
    label_roles: Dict[str, str] | None = None,
    target_specs: Dict[str, TargetSpec] | None = None,
) -> None:
    """
    Validate model-label contract.

    Raises ContractViolation with a clear message if the config is invalid.
    """
    n = len(label_names)

    if contract.mode == "ANY_SINGLE":
        if n != 1:
            raise ContractViolation(
                f"Model '{model_name}' (ANY_SINGLE) requires exactly 1 label, "
                f"but got {n}: {label_names}"
            )

    elif contract.mode == "ANY_MULTI":
        if n < contract.min_labels:
            raise ContractViolation(
                f"Model '{model_name}' (ANY_MULTI) requires at least "
                f"{contract.min_labels} labels, but got {n}: {label_names}"
            )
        if contract.max_labels is not None and n > contract.max_labels:
            raise ContractViolation(
                f"Model '{model_name}' (ANY_MULTI) allows at most "
                f"{contract.max_labels} labels, but got {n}: {label_names}"
            )

    elif contract.mode == "EXACT":
        expected = set(contract.exact_labels)
        actual = set(label_names)
        if expected != actual:
            raise ContractViolation(
                f"Model '{model_name}' (EXACT) requires labels {sorted(expected)}, "
                f"but got {sorted(actual)}"
            )

    elif contract.mode == "ROLE_BASED":
        if label_roles is None:
            raise ContractViolation(
                f"Model '{model_name}' (ROLE_BASED) requires label_roles in config, "
                f"but none were provided"
            )
        for role in contract.required_roles:
            if role not in label_roles:
                raise ContractViolation(
                    f"Model '{model_name}' (ROLE_BASED) requires role '{role}', "
                    f"but label_roles only has: {list(label_roles.keys())}"
                )
            assigned_label = label_roles[role]
            if assigned_label not in label_names:
                raise ContractViolation(
                    f"Model '{model_name}': role '{role}' is mapped to "
                    f"'{assigned_label}', but that label is not in the target list: "
                    f"{label_names}"
                )

    elif contract.mode == "CUSTOM":
        if contract.custom_validator is not None:
            contract.custom_validator(label_names, label_roles, target_specs)
    else:
        raise ContractViolation(
            f"Unknown contract mode '{contract.mode}' for model '{model_name}'"
        )

    logger.info("Preflight contract check passed for model '%s'", model_name)


def validate_labels_exist(
    label_names: List[str],
    available_columns: List[str],
) -> None:
    """Check that all required label/valid/reason columns exist."""
    missing = []
    for name in label_names:
        for prefix in ["label.", "label_valid.", "label_reason."]:
            col = f"{prefix}{name}"
            if col not in available_columns:
                missing.append(col)
    if missing:
        raise ContractViolation(
            f"Panel is missing required label columns: {missing}"
        )
