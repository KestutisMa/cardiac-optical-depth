from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class CaseLayout:
    outputs_root: Path
    case_id: str

    @property
    def case_dir(self) -> Path:
        return self.outputs_root / self.case_id

    @property
    def raw_dir(self) -> Path:
        return self.case_dir / "raw"

    @property
    def derived_dir(self) -> Path:
        return self.case_dir / "derived"

    @property
    def logs_dir(self) -> Path:
        return self.case_dir / "logs"

    @property
    def raw_opencm_dir(self) -> Path:
        return self.raw_dir / "opencm"

    @property
    def raw_opencm_cli_dir(self) -> Path:
        return self.raw_dir / "opencm_cli"

    @property
    def nn_dataset_dir(self) -> Path:
        return self.derived_dir / "nn_dataset"

    @property
    def nn_emission_1d_dir(self) -> Path:
        return self.nn_dataset_dir / "emission_1d"

    @property
    def nn_run_dir(self) -> Path:
        return self.derived_dir / "nn"

    def ensure(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.derived_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.raw_opencm_dir.mkdir(parents=True, exist_ok=True)
        self.raw_opencm_cli_dir.mkdir(parents=True, exist_ok=True)
        self.nn_emission_1d_dir.mkdir(parents=True, exist_ok=True)
        self.nn_run_dir.mkdir(parents=True, exist_ok=True)


def default_outputs_root() -> Path:
    if Path("/outputs").exists():
        return Path("/outputs")
    return Path.cwd() / "outputs"


def new_case_id(prefix: str = "case") -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}"


@dataclass(frozen=True)
class StudyLayout:
    outputs_root: Path
    study_id: str

    @property
    def study_dir(self) -> Path:
        return self.outputs_root / "studies" / self.study_id

    @property
    def cases_root(self) -> Path:
        return self.study_dir / "cases"

    @property
    def derived_dir(self) -> Path:
        return self.study_dir / "derived"

    @property
    def logs_dir(self) -> Path:
        return self.study_dir / "logs"

    @property
    def nn_dataset_dir(self) -> Path:
        return self.derived_dir / "nn_dataset"

    @property
    def nn_run_dir(self) -> Path:
        return self.derived_dir / "nn"

    def ensure(self) -> None:
        self.cases_root.mkdir(parents=True, exist_ok=True)
        self.derived_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.nn_dataset_dir.mkdir(parents=True, exist_ok=True)
        self.nn_run_dir.mkdir(parents=True, exist_ok=True)
