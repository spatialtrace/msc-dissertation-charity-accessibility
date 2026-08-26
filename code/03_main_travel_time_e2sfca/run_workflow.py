from __future__ import annotations

import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent


def main() -> None:
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(RUN_DIR / "scripts"))
    from e2sfca_workflow import E2SFCAWorkflow

    workflow = E2SFCAWorkflow(RUN_DIR)
    workflow.load_and_audit_inputs()
    workflow.load_common_spatial_foundation()
    workflow.load_provider_capacity_and_demand()
    workflow.calculate_od_impedance()
    workflow.e2sfca_step1()
    workflow.e2sfca_step2()
    workflow.standardise_accessibility()
    workflow.write_descriptive_outputs()
    workflow.run_final_qa()
    workflow.prepare_longitudinal_handoff()


if __name__ == "__main__":
    main()
