#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


@dataclasses.dataclass
class Config:
    project_root: Path = dataclasses.field(default_factory=lambda: Path.cwd())
    train_cmd: List[str] = dataclasses.field(default_factory=lambda: ["python", "train.py"])

    test_cmd: List[str] = dataclasses.field(
        default_factory=lambda: shlex.split(os.environ.get("PHOENIX_TEST_CMD", ""))
    )

    primary_llm: str = dataclasses.field(default_factory=lambda: os.environ.get("PHOENIX_PRIMARY", "gemini_cli"))
    max_retries_per_signature: int = int(os.environ.get("PHOENIX_MAX_RETRIES", "3"))
    log_tail_lines: int = int(os.environ.get("PHOENIX_TAIL", "200"))
    dry_run: bool = os.environ.get("PHOENIX_DRY_RUN", "0") == "1"

    state_path: Path = dataclasses.field(default_factory=lambda: Path(".phoenix_cli/state.json"))
    backups_dir: Path = dataclasses.field(default_factory=lambda: Path(".phoenix_cli/backups"))
    log_path: Path = dataclasses.field(default_factory=lambda: Path(".phoenix_cli/run.log"))

    allow_modify_globs: Tuple[str, ...] = ("*.py",)
    deny_dirs: Tuple[str, ...] = (".git", "venv", ".venv", "__pycache__", ".phoenix_cli")
    allowlist_files: Tuple[str, ...] = dataclasses.field(
        default_factory=lambda: tuple(filter(None, os.environ.get("PHOENIX_ALLOWLIST", "").split(";")))
    )

    max_file_chars: int = int(os.environ.get("PHOENIX_MAX_FILE_CHARS", "120000"))
    max_prompt_chars: int = int(os.environ.get("PHOENIX_MAX_PROMPT_CHARS", "180000"))

    gemini_cli_bin: str = dataclasses.field(default_factory=lambda: os.environ.get("GEMINI_CLI_BIN", "gemini"))

    curl_bin: str = dataclasses.field(default_factory=lambda: os.environ.get("CURL_BIN", "curl"))
    gemini_api_key_env: str = dataclasses.field(default_factory=lambda: os.environ.get("GEMINI_API_KEY_ENV", "GEMINI_API_KEY"))
    gemini_model: str = dataclasses.field(default_factory=lambda: os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"))
    request_timeout_s: int = int(os.environ.get("PHOENIX_HTTP_TIMEOUT", "120"))

    python_bin: str = dataclasses.field(default_factory=lambda: sys.executable)

    heartbeat_timeout_min: int = int(os.environ.get("PHOENIX_HEARTBEAT_MIN", "15"))
    redact_patterns: Tuple[str, ...] = (
        r"(?i)api_key\s*[=:]\s*['\"]?[-A-Za-z0-9_\.]{8,}['\"]?",
        r"(?i)token\s*[=:]\s*['\"]?[-A-Za-z0-9_\.]{8,}['\"]?",
        r"(?i)bearer\s+[A-Za-z0-9._-]{8,}",
        r"(?i)secret\s*[=:]\s*['\"]?[-A-Za-z0-9_\.]{6,}['\"]?",
    )
    log_max_bytes: int = int(os.environ.get("PHOENIX_LOG_MAX_BYTES", str(5 * 1024 * 1024)))
    max_backups_per_file: int = int(os.environ.get("PHOENIX_MAX_BACKUPS", "20"))
    cooldown_seconds_on_stop: int = int(os.environ.get("PHOENIX_COOLDOWN_SECONDS", "0"))

    def ensure_dirs(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)


class Logger:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()

    def _rotate_if_needed(self) -> None:
        try:
            if not self.cfg.log_path.exists():
                return
            if self.cfg.log_path.stat().st_size <= self.cfg.log_max_bytes:
                return
            rotated = self.cfg.log_path.with_suffix(self.cfg.log_path.suffix + ".1")
            if rotated.exists():
                rotated.unlink()
            self.cfg.log_path.rename(rotated)
        except Exception:
            pass

    def log(self, msg: str) -> None:
        line = f"[{_now()}] {msg}"
        with self._lock:
            self._rotate_if_needed()
            print(line, flush=True)
            try:
                with self.cfg.log_path.open("a", encoding="utf-8", errors="ignore") as f:
                    f.write(line + "\n")
            except Exception:
                pass


class Redactor:
    def __init__(self, patterns: Tuple[str, ...]) -> None:
        self.patterns = [re.compile(p) for p in patterns]

    def redact(self, text: str) -> str:
        redacted = text
        for pattern in self.patterns:
            redacted = pattern.sub("<REDACTED>", redacted)
        return redacted


class RollingBuffer:
    def __init__(self, max_lines: int) -> None:
        self.max_lines = max_lines
        self._lines: List[str] = []
        self._lock = threading.Lock()

    def add(self, line: str) -> None:
        with self._lock:
            self._lines.append(line.rstrip("\n"))
            if len(self._lines) > self.max_lines:
                self._lines = self._lines[-self.max_lines :]

    def tail_text(self) -> str:
        with self._lock:
            return "\n".join(self._lines)


class StateStore:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self.cfg.state_path.exists():
            try:
                self._state = json.loads(self.cfg.state_path.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                self._state = {}
        if "retries" not in self._state:
            self._state["retries"] = {}

    def _save(self) -> None:
        tmp = self.cfg.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.cfg.state_path)

    def get_retry(self, signature: str) -> int:
        with self._lock:
            return int(self._state.get("retries", {}).get(signature, 0))

    def inc_retry(self, signature: str) -> int:
        with self._lock:
            self._state.setdefault("retries", {})
            self._state["retries"][signature] = int(self._state["retries"].get(signature, 0)) + 1
            self._save()
            return int(self._state["retries"][signature])


class GeminiCLIClient:
    def __init__(self, cfg: Config, logger: Logger) -> None:
        self.cfg = cfg
        self.logger = logger

    def request_fix(self, prompt: str) -> Dict[str, Any]:
        cmd = [self.cfg.gemini_cli_bin, "-p", prompt]
        self.logger.log(f"GeminiCLI call: {' '.join(cmd[:3])} <prompt>")
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        if proc.returncode != 0:
            raise RuntimeError(f"Gemini CLI failed rc={proc.returncode} stderr={(proc.stderr or '')[-1000:]}")
        return _extract_json_object(proc.stdout)


class GeminiApiCurlClient:
    def __init__(self, cfg: Config, logger: Logger) -> None:
        self.cfg = cfg
        self.logger = logger

    def request_fix(self, prompt: str, target_path: str) -> Dict[str, Any]:
        api_key = os.environ.get(self.cfg.gemini_api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing env {self.cfg.gemini_api_key_env}")

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.cfg.gemini_model}:generateContent?key={api_key}"
        )

        patch_schema = {
            "type": "OBJECT",
            "properties": {
                "file_path": {"type": "STRING"},
                "mode": {"type": "STRING"},
                "diff": {"type": "STRING"},
                "start_line": {"type": "NUMBER"},
                "end_line": {"type": "NUMBER"},
                "code": {"type": "STRING"},
            },
            "required": ["file_path", "mode"],
        }

        schema = {
            "type": "OBJECT",
            "properties": {
                "patches": {
                    "type": "ARRAY",
                    "items": patch_schema,
                }
            },
            "required": ["patches"],
        }

        body = {
            "system_instruction": {
                "parts": [
                    {
                        "text": (
                            "Return JSON only with top-level patches array."
                            "Each patch must include file_path, mode (unified_diff or replace_range),"
                            "and required fields per mode."
                            "No markdown or commentary."
                        )
                    }
                ]
            },
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "response_mime_type": "application/json",
                "response_schema": schema,
            },
        }

        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")

        cmd = [
            self.cfg.curl_bin,
            "-sS",
            "-H",
            "Content-Type: application/json",
            "-X",
            "POST",
            url,
            "--data-binary",
            "@-",
        ]

        self.logger.log(f"GeminiAPI curl call model={self.cfg.gemini_model} target={target_path}")
        proc = subprocess.run(
            cmd,
            input=payload,
            capture_output=True,
            timeout=self.cfg.request_timeout_s,
        )
        if proc.returncode != 0:
            err = (proc.stderr or b"").decode("utf-8", errors="ignore")
            raise RuntimeError(f"curl failed rc={proc.returncode} stderr={err[-1200:]}")

        outer = json.loads((proc.stdout or b"{}").decode("utf-8", errors="ignore"))
        text = _extract_gemini_text(outer)
        return _extract_json_object(text)


def _extract_gemini_text(resp: Dict[str, Any]) -> str:
    parts: List[str] = []
    for candidate in resp.get("candidates", []) or []:
        content = (candidate or {}).get("content", {}) or {}
        for part in content.get("parts", []) or []:
            text = (part or {}).get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts).strip()


def _extract_json_object(raw: str) -> Dict[str, Any]:
    data = raw.strip()
    if not data:
        raise RuntimeError("Empty LLM output")

    try:
        return json.loads(data)
    except Exception:
        pass

    match = re.search(r"\{.*\}", data, flags=re.DOTALL)
    if not match:
        raise RuntimeError(f"Failed to locate JSON object head={data[:200]!r}")
    return json.loads(match.group(0))


class PatchApplier:
    def __init__(self, cfg: Config, logger: Logger) -> None:
        self.cfg = cfg
        self.logger = logger

    def is_safe_target(self, path: Path) -> bool:
        try:
            rel = path.resolve().relative_to(self.cfg.project_root.resolve())
        except Exception:
            return False

        for part in rel.parts:
            if part in self.cfg.deny_dirs:
                return False
        if self.cfg.allowlist_files:
            if not any(rel.match(pat) for pat in self.cfg.allowlist_files):
                return False

        return any(rel.match(glob) or path.match(glob) for glob in self.cfg.allow_modify_globs)

    def _prune_backups(self, path: Path) -> None:
        backups = sorted(
            self.cfg.backups_dir.glob(f"{path.name}.*.bak"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old in backups[self.cfg.max_backups_per_file :]:
            try:
                old.unlink()
            except Exception:
                pass

    def backup_and_write(self, path: Path, new_code: str, original: str) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup = self.cfg.backups_dir / f"{path.name}.{timestamp}.bak"
        backup.write_text(original, encoding="utf-8")
        self._prune_backups(path)
        if self.cfg.dry_run:
            self.logger.log(f"DRY_RUN would write {path}")
            return backup

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(new_code, encoding="utf-8")
        try:
            with tmp_path.open("r+", encoding="utf-8", errors="ignore") as f:
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            pass
        tmp_path.replace(path)
        return backup

    def py_compile(self, path: Path) -> Tuple[bool, str]:
        cmd = [self.cfg.python_bin, "-m", "py_compile", str(path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        return proc.returncode == 0, (proc.stderr or "")[-2000:]

    def run_test_cmd(self) -> Tuple[bool, str]:
        if not self.cfg.test_cmd:
            return True, ""
        proc = subprocess.run(
            self.cfg.test_cmd,
            cwd=str(self.cfg.project_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        output = "\n".join([
            (proc.stdout or "").strip(),
            (proc.stderr or "").strip(),
        ]).strip()
        return proc.returncode == 0, output

    def rollback_all(self, backups: Dict[Path, Path]) -> None:
        for path, backup in backups.items():
            try:
                if self.cfg.dry_run:
                    self.logger.log(f"DRY_RUN would rollback {path} from {backup}")
                    continue
                path.write_text(backup.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
            except Exception:
                pass

    def _apply_replace_range(self, content: str, start: int, end: int, code: str) -> str:
        lines = content.splitlines(keepends=True)
        start_idx = max(start - 1, 0)
        end_idx = max(end, 0)
        if start_idx > len(lines):
            start_idx = len(lines)
        if end_idx > len(lines):
            end_idx = len(lines)
        replacement = code.splitlines(keepends=True)
        return "".join(lines[:start_idx] + replacement + lines[end_idx:])

    def _apply_unified_diff(self, content: str, diff: str) -> str:
        original_lines = content.splitlines(keepends=True)
        diff_lines = diff.splitlines(keepends=True)
        patched_lines: List[str] = []
        idx = 0
        i = 0

        while i < len(diff_lines):
            line = diff_lines[i]
            if not line.startswith("@@"):
                i += 1
                continue
            match = re.match(r"@@ -(?P<l1>\d+)(,(?P<n1>\d+))? \+(?P<l2>\d+)(,(?P<n2>\d+))? @@", line)
            if not match:
                raise ValueError("Invalid unified diff header")
            start_old = int(match.group("l1")) - 1
            while idx < start_old and idx < len(original_lines):
                patched_lines.append(original_lines[idx])
                idx += 1
            i += 1
            while i < len(diff_lines) and not diff_lines[i].startswith("@@"):
                hunk_line = diff_lines[i]
                if hunk_line.startswith(" "):
                    if idx >= len(original_lines):
                        raise ValueError("Context line out of range")
                    patched_lines.append(original_lines[idx])
                    idx += 1
                elif hunk_line.startswith("-"):
                    idx += 1
                elif hunk_line.startswith("+"):
                    patched_lines.append(hunk_line[1:])
                else:
                    raise ValueError("Unknown diff line prefix")
                i += 1
        patched_lines.extend(original_lines[idx:])
        return "".join(patched_lines)

    def apply_patch_set(self, patches: List[Dict[str, Any]]) -> Tuple[bool, str]:
        backups: Dict[Path, Path] = {}
        originals: Dict[Path, str] = {}
        new_contents: Dict[Path, str] = {}

        for patch in patches:
            file_path = Path(str(patch.get("file_path"))).resolve()
            if not self.is_safe_target(file_path):
                raise RuntimeError(f"Unsafe target path {file_path}")
            mode = patch.get("mode")
            originals.setdefault(file_path, file_path.read_text(encoding="utf-8", errors="ignore") if file_path.exists() else "")
            updated = new_contents.get(file_path, originals[file_path])

            if mode == "replace_range":
                start = int(patch.get("start_line", 0))
                end = int(patch.get("end_line", 0))
                code = patch.get("code")
                if not isinstance(code, str):
                    raise RuntimeError("replace_range requires code")
                updated = self._apply_replace_range(updated, start, end, code)
            elif mode == "unified_diff":
                diff = patch.get("diff")
                if not isinstance(diff, str):
                    raise RuntimeError("unified_diff requires diff")
                updated = self._apply_unified_diff(updated, diff)
            else:
                raise RuntimeError(f"Unknown patch mode {mode}")

            new_contents[file_path] = updated

        try:
            for path, content in new_contents.items():
                backup = self.backup_and_write(path, content, originals[path])
                backups[path] = backup

            compile_errors: List[str] = []
            for path in new_contents:
                if path.suffix == ".py":
                    ok, err = self.py_compile(path)
                    if not ok:
                        compile_errors.append(f"{path}: {err}")
            if compile_errors:
                raise RuntimeError("; ".join(compile_errors))

            test_ok, test_output = self.run_test_cmd()
            if not test_ok:
                raise RuntimeError(f"test_cmd failed: {test_output[-800:]}" if test_output else "test_cmd failed")
        except Exception as exc:
            self.logger.log(f"apply_patch_set error: {exc}")
            self.rollback_all(backups)
            return False, str(exc)

        return True, ""


class ErrorHandler:
    TB_RE = re.compile(r'File "([^"]+)", line (\d+), in ')

    def __init__(self, cfg: Config, logger: Logger, state: StateStore) -> None:
        self.cfg = cfg
        self.logger = logger
        self.state = state
        self.patch = PatchApplier(cfg, logger)
        self.gemini_cli = GeminiCLIClient(cfg, logger)
        self.gemini_api = GeminiApiCurlClient(cfg, logger)
        self.redactor = Redactor(cfg.redact_patterns)

    def _select_target_from_traceback(self, stderr_tail: str) -> Tuple[Optional[Path], Optional[int]]:
        matches = list(self.TB_RE.finditer(stderr_tail))
        root = self.cfg.project_root.resolve()
        for match in reversed(matches):
            candidate = Path(match.group(1)).resolve()
            if candidate.suffix != ".py":
                continue
            try:
                candidate.relative_to(root)
            except Exception:
                continue
            if not self.patch.is_safe_target(candidate):
                continue
            return candidate, int(match.group(2))
        return None, None

    def _fallback_target(self) -> Path:
        for arg in self.cfg.train_cmd:
            if arg.endswith(".py"):
                return (self.cfg.project_root / arg).resolve()
        return (self.cfg.project_root / "train.py").resolve()

    def _read_file_limited(self, path: Path) -> str:
        content = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
        if len(content) > self.cfg.max_file_chars:
            content = content[: self.cfg.max_file_chars] + "\n\n# <truncated>"
        return self.redactor.redact(content)

    def _sanitize_log(self, text: str) -> str:
        clean = re.sub(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}", "<TS>", text)
        clean = re.sub(r"\d{2}:\d{2}:\d{2}", "<TS>", clean)
        clean = re.sub(r"\r", "", clean)
        clean = re.sub(r"\d+%", "<PCT>", clean)
        return self.redactor.redact(clean)

    def _build_prompt(
        self,
        target: Path,
        line_no: Optional[int],
        stderr_tail: str,
        stdout_tail: str,
        previous_error: str = "",
    ) -> str:
        content = self._read_file_limited(target)
        stderr_safe = self._sanitize_log(stderr_tail)
        stdout_safe = self._sanitize_log(stdout_tail)
        allowlist = ";".join(self.cfg.allowlist_files) if self.cfg.allowlist_files else ""
        parts = [
            "You are a surgical Python bugfixer.",
            "Output JSON only with a top-level 'patches' array.",
            "Each patch object must include: file_path, mode, and depending on mode:",
            "- mode: unified_diff => diff: unified diff content",
            "- mode: replace_range => start_line, end_line (1-indexed, inclusive), code",
            "Rules:",
            "- file_path must stay inside project_root and match the provided targets only.",
            "- Keep changes minimal to fix the error; avoid refactors or formatting.",
            "- No markdown, no explanations, JSON only.",
            "- Do not add dependencies; standard library only.",
            f"project_root={self.cfg.project_root}",
            f"allow_modify_globs={self.cfg.allow_modify_globs}",
            f"deny_dirs={self.cfg.deny_dirs}",
            f"allowlist_files={allowlist}",
            "",
            "Context",
            f"OS: {os.name}",
            f"Python: {sys.version.split()[0]}",
            "",
            "Training command",
            " ".join(self.cfg.train_cmd),
            "",
            "stderr tail (redacted)",
            stderr_safe,
            "",
            "stdout tail (redacted)",
            stdout_safe,
        ]
        if line_no is not None:
            parts += ["", f"Target line: {line_no}"]
        if previous_error:
            parts += ["", "Previous validation/test error", self._sanitize_log(previous_error)]
        parts += ["", "Target file content (redacted)", content]
        prompt = "\n".join(parts)
        if len(prompt) > self.cfg.max_prompt_chars:
            prompt = prompt[: self.cfg.max_prompt_chars] + "\n\n<truncated>"
        return prompt

    def _extract_exception_info(self, stderr_tail: str) -> Tuple[str, str]:
        exc_name = ""
        exc_msg = ""
        for line in reversed(stderr_tail.splitlines()):
            if ":" in line and re.match(r"[A-Za-z_][A-Za-z0-9_]*:\s", line.strip()):
                parts = line.split(":", 1)
                exc_name = parts[0].strip()
                exc_msg = parts[1].strip()
                break
        tb_matches = list(self.TB_RE.finditer(stderr_tail))
        last_tb = tb_matches[-1].group(0) if tb_matches else ""
        return exc_name + " " + exc_msg, last_tb

    def _error_signature(self, exit_code: int, stderr_tail: str, target: Path) -> str:
        sanitized = self._sanitize_log(stderr_tail)
        exc_line, last_tb = self._extract_exception_info(sanitized)
        head = "\n".join((sanitized or "").splitlines()[:20])
        base = f"rc={exit_code}\nfile={target.name}\n{exc_line}\n{last_tb}\n{head}"
        return _sha256(base)

    def handle_failure(self, exit_code: int, stderr_tail: str, stdout_tail: str) -> bool:
        target, line_no = self._select_target_from_traceback(stderr_tail)
        if target is None:
            target = self._fallback_target()

        signature = self._error_signature(exit_code, stderr_tail, target)
        attempt = self.state.get_retry(signature)
        if attempt >= self.cfg.max_retries_per_signature:
            self.logger.log(f"STOP retries exceeded signature={signature} target={target}")
            return False

        previous_error = ""
        for _ in range(2):
            current_attempt = self.state.get_retry(signature) + 1
            self.logger.log(
                f"FAIL rc={exit_code} target={target} signature={signature} attempt={current_attempt}"
            )
            prompt = self._build_prompt(target, line_no, stderr_tail, stdout_tail, previous_error)
            data = self._call_llm(prompt, target)
            patches = self._validate_response(data, target)

            ok, err = self.patch.apply_patch_set(patches)
            self.state.inc_retry(signature)
            if ok:
                touched = {Path(p["file_path"]).name for p in patches}
                self.logger.log(f"PATCH OK files={','.join(sorted(touched))}")
                return True

            previous_error = err
            self.logger.log(f"PATCH FAILED err={err}")
            if self.state.get_retry(signature) >= self.cfg.max_retries_per_signature:
                self.logger.log(f"STOP after failures signature={signature}")
                return False

        return False

    def _call_llm(self, prompt: str, target: Path) -> Dict[str, Any]:
        if self.cfg.primary_llm == "gemini_cli":
            return self.gemini_cli.request_fix(prompt)
        if self.cfg.primary_llm == "gemini_api":
            return self.gemini_api.request_fix(prompt, str(target))
        if self.cfg.primary_llm == "copilot":
            raise RuntimeError("Copilot CLI is not enabled by default. Set primary_llm accordingly if implemented.")
        raise RuntimeError(f"Unknown PHOENIX_PRIMARY={self.cfg.primary_llm}")

    def _validate_response(self, data: Dict[str, Any], target: Path) -> List[Dict[str, Any]]:
        patches = data.get("patches")
        if not isinstance(patches, list) or not patches:
            raise RuntimeError("LLM JSON must contain non-empty patches list")

        normalized: List[Dict[str, Any]] = []
        for patch in patches:
            if not isinstance(patch, dict):
                continue
            file_path = patch.get("file_path")
            mode = patch.get("mode")
            if not isinstance(file_path, str) or not isinstance(mode, str):
                continue
            resolved = Path(file_path).resolve()
            if not self.patch.is_safe_target(resolved):
                raise RuntimeError(f"Unsafe target path {resolved}")
            if mode not in {"unified_diff", "replace_range"}:
                raise RuntimeError(f"Unknown patch mode {mode}")
            normalized_patch = {"file_path": str(resolved), "mode": mode}
            if mode == "unified_diff":
                diff = patch.get("diff")
                if not isinstance(diff, str):
                    raise RuntimeError("unified_diff requires diff")
                normalized_patch["diff"] = diff
            else:
                start_line = patch.get("start_line")
                end_line = patch.get("end_line")
                code = patch.get("code")
                if not (isinstance(start_line, int) and isinstance(end_line, int) and isinstance(code, str)):
                    raise RuntimeError("replace_range requires start_line, end_line, code")
                normalized_patch.update({"start_line": start_line, "end_line": end_line, "code": code})
            normalized.append(normalized_patch)

        if not normalized:
            raise RuntimeError("No valid patches returned")

        if self.cfg.allowlist_files:
            pass
        else:
            for patch in normalized:
                if Path(patch["file_path"]).resolve() != target.resolve():
                    raise RuntimeError("Patches must target the primary file unless allowlist is set")

        return normalized


class ProcessManager:
    def __init__(self, cfg: Config, logger: Logger) -> None:
        self.cfg = cfg
        self.logger = logger

    def _start_process(self) -> subprocess.Popen:
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        return subprocess.Popen(
            self.cfg.train_cmd,
            cwd=str(self.cfg.project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            start_new_session=os.name != "nt",
            creationflags=creationflags,
        )

    def _pump_stream(self, stream: Any, buffer: RollingBuffer, to_stderr: bool, heartbeat: Dict[str, float], lock: threading.Lock) -> None:
        try:
            for line in iter(stream.readline, ""):
                with lock:
                    heartbeat["last"] = time.time()
                buffer.add(line)
                if to_stderr:
                    sys.stderr.write(line)
                    sys.stderr.flush()
                else:
                    sys.stdout.write(line)
                    sys.stdout.flush()
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def _kill_process_tree(self, proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
            else:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    proc.kill()
        finally:
            try:
                proc.wait(timeout=10)
            except Exception:
                pass

    def run_training(self) -> Tuple[int, str, str]:
        self.logger.log(f"RUN {' '.join(self.cfg.train_cmd)}")
        proc = self._start_process()

        out_buf = RollingBuffer(self.cfg.log_tail_lines)
        err_buf = RollingBuffer(self.cfg.log_tail_lines)

        heartbeat = {"last": time.time()}
        hb_lock = threading.Lock()
        stop_event = threading.Event()

        t_out = threading.Thread(
            target=self._pump_stream, args=(proc.stdout, out_buf, False, heartbeat, hb_lock), daemon=True
        )
        t_err = threading.Thread(
            target=self._pump_stream, args=(proc.stderr, err_buf, True, heartbeat, hb_lock), daemon=True
        )
        t_out.start()
        t_err.start()

        def watchdog() -> None:
            timeout = self.cfg.heartbeat_timeout_min * 60
            if timeout <= 0:
                return
            while not stop_event.is_set():
                time.sleep(5)
                if proc.poll() is not None:
                    return
                with hb_lock:
                    idle = time.time() - heartbeat["last"]
                if idle >= timeout:
                    self.logger.log(
                        f"HEARTBEAT timeout {idle:.1f}s without output -> kill process tree"
                    )
                    self._kill_process_tree(proc)
                    stop_event.set()
                    return

        watchdog_thread = threading.Thread(target=watchdog, daemon=True)
        watchdog_thread.start()

        rc = proc.wait()
        stop_event.set()
        t_out.join(timeout=2)
        t_err.join(timeout=2)
        watchdog_thread.join(timeout=2)
        self.logger.log(f"EXIT rc={rc}")

        if rc != 0:
            self._kill_process_tree(proc)

        return rc, err_buf.tail_text(), out_buf.tail_text()


def main() -> int:
    cfg = Config()
    cfg.ensure_dirs()
    logger = Logger(cfg)
    state = StateStore(cfg)
    handler = ErrorHandler(cfg, logger, state)
    process_manager = ProcessManager(cfg, logger)

    while True:
        exit_code, stderr_tail, stdout_tail = process_manager.run_training()
        if exit_code == 0:
            logger.log("DONE training finished successfully")
            return 0

        try:
            fixed = handler.handle_failure(exit_code, stderr_tail, stdout_tail)
        except Exception as exc:  # pragma: no cover - defensive log path
            logger.log(f"FATAL during handle_failure: {exc}")
            return 2

        if not fixed:
            if cfg.cooldown_seconds_on_stop > 0:
                logger.log(
                    f"COOLDOWN waiting {cfg.cooldown_seconds_on_stop}s before stopping after repeated failures"
                )
                time.sleep(cfg.cooldown_seconds_on_stop)
            logger.log("STOP unable to auto fix")
            return 1

        logger.log("RETRY restarting training after patch")


if __name__ == "__main__":
    raise SystemExit(main())
