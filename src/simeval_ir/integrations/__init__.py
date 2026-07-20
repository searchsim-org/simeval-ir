"""Integration helpers for popular third-party frameworks.

These modules are kept lazy so that importing ``simeval_ir`` does not pull
in heavy or optional dependencies (PyTerrier, sentence-transformers, etc.).
Import the specific helper directly:

    from simeval_ir.integrations.simiir import from_simiir_logs
    from simeval_ir.integrations.sim4ia_bench import load_task_a, load_task_b
    from simeval_ir.integrations.pyterrier import from_pyterrier_experiment
"""

__all__: list[str] = []
