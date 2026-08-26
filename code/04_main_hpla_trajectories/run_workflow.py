from __future__ import annotations

import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent


def main() -> None:
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(RUN_DIR / "scripts"))
    from downstream_workflow import AccessibilityLongitudinalWorkflow

    workflow = AccessibilityLongitudinalWorkflow(RUN_DIR)
    workflow.load_and_audit_inputs()
    workflow.load_accessibility_outputs()
    workflow.write_accessibility_statistics()
    workflow.make_accessibility_maps()
    workflow.write_changes()
    workflow.write_icb_summary()
    workflow.classify_annual_hpla()
    workflow.build_trajectories()
    workflow.write_transition_statistics()
    workflow.final_qa_and_method()


if __name__ == "__main__":
    main()
