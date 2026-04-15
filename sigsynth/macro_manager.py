from __future__ import annotations

from pathlib import Path

import yaml

from sigsynth.models import AppConfig


class MacroManager:
    def __init__(self, macro_dir: Path | str = "macros") -> None:
        self.macro_dir = Path(macro_dir)
        self.macro_dir.mkdir(parents=True, exist_ok=True)

    def list_macros(self) -> list[str]:
        return sorted(path.name for path in self.macro_dir.glob("*.yaml"))

    def load(self, macro_name: str) -> AppConfig:
        macro_path = self.macro_dir / macro_name
        with macro_path.open("r", encoding="utf-8") as fp:
            payload = yaml.safe_load(fp) or {}
        return AppConfig.from_dict(payload)

    def save(self, macro_name: str, config: AppConfig) -> Path:
        path = self.macro_dir / macro_name
        with path.open("w", encoding="utf-8") as fp:
            yaml.safe_dump(config.to_dict(), fp, sort_keys=False)
        return path
