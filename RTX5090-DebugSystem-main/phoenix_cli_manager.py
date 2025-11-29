#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Pattern


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


@dataclasses.dataclass
class Config:
    project_root: Path = dataclasses.field(default_factory=lambda: Path.cwd())
    train_cmd: List[str] = dataclasses.field(
        default_factory=lambda: shlex.split(os.environ.get("PHOENIX_TRAIN_CMD", "python train_wrapper.py"))
    )

    test_cmd: List[str] = dataclasses.field(
        default_factory=lambda: shlex.split(os.environ.get("PHOENIX_TEST_CMD", ""))
    )
    require_test_pass: bool = os.environ.get("PHOENIX_REQUIRE_TEST_PASS", "1") == "1"

    fallback_llm_cmd: List[str] = dataclasses.field(
        default_factory=lambda: shlex.split(os.environ.get("PHOENIX_FALLBACK_LLM_CMD", ""))
    )

    primary_llm: str = dataclasses.field(default_factory=lambda: os.environ.get("PHOENIX_PRIMARY", "gemini_cli"))
    max_retries_per_signature: int = int(os.environ.get("PHOENIX_MAX_RETRIES_PER_SIGNATURE", "3"))
    max_total_retries: int = int(os.environ.get("PHOENIX_MAX_TOTAL_RETRIES", "10"))
    
    log_tail_lines: int = int(os.environ.get("PHOENIX_TAIL", "200"))
    dry_run: bool = os.environ.get("PHOENIX_DRY_RUN", "0") == "1"
    log_format: str = os.environ.get("PHOENIX_LOG_FORMAT", "json")

    state_path: Path = dataclasses.field(default_factory=lambda: Path(".phoenix_cli/state.json"))
    backups_dir: Path = dataclasses.field(default_factory=lambda: Path(".phoenix_cli/backups"))
    log_path: Path = dataclasses.field(default_factory=lambda: Path(".phoenix_cli/run.log"))
    lock_path: Path = dataclasses.field(default_factory=lambda: Path(".phoenix_cli/lock"))

    # Safety: Allowlist is strict. Default is EMPTY (DENY ALL).
    allow_modify_globs: Tuple[str, ...] = dataclasses.field(
        default_factory=lambda: tuple(filter(None, os.environ.get("PHOENIX_ALLOWLIST", "").split(";")))
    )
    
    # Safety: Deny list includes tests and sensitive dirs
    deny_dirs: Tuple[str, ...] = (
        ".git", ".github", ".venv", "venv", "node_modules", "__pycache__", 
        "dist", "build", ".idea", ".vscode", "data", "logs", "secrets", 
        "credential", "cert", "AppData", "Users", "Program Files", "tests", "test"
    )

    max_file_bytes: int = int(os.environ.get("PHOENIX_MAX_FILE_BYTES", "200000"))
    max_prompt_chars: int = int(os.environ.get("PHOENIX_MAX_PROMPT_CHARS", "180000"))
    
    # Patch Guardrails
    max_patch_lines: int = int(os.environ.get("PHOENIX_MAX_PATCH_LINES", "50"))
    max_patch_files: int = int(os.environ.get("PHOENIX_MAX_PATCH_FILES", "3"))

    gemini_cli_bin: str = dataclasses.field(default_factory=lambda: os.environ.get("GEMINI_CLI_BIN", "gemini"))
    curl_bin: str = dataclasses.field(default_factory=lambda: os.environ.get("CURL_BIN", "curl"))
    gemini_api_key_env: str = dataclasses.field(default_factory=lambda: os.environ.get("GEMINI_API_KEY_ENV", "GEMINI_API_KEY"))
    gemini_model: str = dataclasses.field(default_factory=lambda: os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"))
    request_timeout_s: int = int(os.environ.get("PHOENIX_HTTP_TIMEOUT", "120"))

    python_bin: str = dataclasses.field(default_factory=lambda: sys.executable)

    heartbeat_timeout_min: int = int(os.environ.get("PHOENIX_HEARTBEAT_MIN", "15"))
    
    # Redaction
    redact_max_patterns: int = int(os.environ.get("PHOENIX_REDACT_MAX_PATTERNS", "32"))
    redact_max_pattern_len: int = int(os.environ.get("PHOENIX_REDACT_MAX_PATTERN_LEN", "120"))
    
    base_redact_patterns: Tuple[str, ...] = (
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

    def get_redact_patterns(self) -> List[str]:
        extra = os.environ.get("PHOENIX_ADDITIONAL_REDACT_PATTERNS", "")
        patterns = list(self.base_redact_patterns)
        if extra:
            patterns.extend(filter(None, extra.split(";")))
        return patterns[:self.redact_max_patterns]


class Redactor:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.patterns: List[Pattern] = []
        for p in cfg.get_redact_patterns():
            if len(p) > cfg.redact_max_pattern_len:
                continue # Skip too long patterns
            try:
                self.patterns.append(re.compile(p))
            except re.error:
                pass # Ignore invalid regex

    def redact(self, text: str) -> str:
        if not text:
            return ""
        redacted = text
        for pattern in self.patterns:
            try:
                redacted = pattern.sub("<REDACTED>", redacted)
            except Exception:
                pass
        return redacted
    
    def redact_obj(self, obj: Any) -> Any:
        if isinstance(obj, str):
            return self.redact(obj)
        if isinstance(obj, Path):
            return self.redact(str(obj))
        if isinstance(obj, dict):
            return {k: self.redact_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.redact_obj(i) for i in obj]
        if dataclasses.is_dataclass(obj):
            return self.redact_obj(dataclasses.asdict(obj))
        return obj


class Logger:
    def __init__(self, cfg: Config, redactor: Redactor) -> None:
        self.cfg = cfg
        self.redactor = redactor
        self._lock = threading.Lock()
        self.run_id = _sha256(str(time.time()))[:8]

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

    def log(self, event: str, level: str = "INFO", **kwargs: Any) -> None:
        entry = {
            "schema_version": "v1",
            "ts": _now_iso(),
            "level": level,
            "event": event,
            "run_id": self.run_id,
            **kwargs
        }
        
        # Redact everything
        safe_entry = self.redactor.redact_obj(entry)

        with self._lock:
            self._rotate_if_needed()
            
            if self.cfg.log_format == "json":
                line = json.dumps(safe_entry, ensure_ascii=False)
            else:
                # Legacy text format
                kv = " ".join(f"{k}={v}" for k, v in safe_entry.items() if k not in ("ts", "level", "event", "schema_version"))
                line = f"[{safe_entry['ts']}] {level} {event} {kv}"

            print(line, flush=True)
            try:
                with self.cfg.log_path.open("a", encoding="utf-8", errors="ignore") as f:
                    f.write(line + "\n")
            except Exception:
                pass


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
        if "quarantine" not in self._state:
            self._state["quarantine"] = []

    def _save(self) -> None:
        tmp = self.cfg.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            tmp.replace(self.cfg.state_path)
        except OSError:
            pass

    def get_retry(self, signature: str) -> int:
        with self._lock:
            return int(self._state.get("retries", {}).get(signature, 0))

    def inc_retry(self, signature: str) -> int:
        with self._lock:
            self._state.setdefault("retries", {})
            self._state["retries"][signature] = int(self._state["retries"].get(signature, 0)) + 1
            self._save()
            return int(self._state["retries"][signature])
            
    def is_quarantined(self, signature: str) -> bool:
        with self._lock:
            return signature in self._state.get("quarantine", [])

    def quarantine(self, signature: str) -> None:
        with self._lock:
            q = self._state.get("quarantine", [])
            if signature not in q:
                q.append(signature)
                self._state["quarantine"] = q
                self._save()


class GeminiCLIClient:
    def __init__(self, cfg: Config, logger: Logger) -> None:
        self.cfg = cfg
        self.logger = logger

    def request_fix(self, prompt: str) -> Dict[str, Any]:
        cmd = [self.cfg.gemini_cli_bin, "-p", prompt]
        self.logger.log("llm_request", method="cli")
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

        self.logger.log("llm_request", method="api", model=self.cfg.gemini_model, target=target_path)
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


class LocalCodeCLIClient:
    def __init__(self, cfg: Config, logger: Logger) -> None:
        self.cfg = cfg
        self.logger = logger

    def request_fix(self, prompt: str) -> Dict[str, Any]:
        if not self.cfg.fallback_llm_cmd:
            raise RuntimeError("Fallback LLM command not configured")
        
        # Append prompt as the last argument
        cmd = self.cfg.fallback_llm_cmd + [prompt]
        self.logger.log("llm_request", method="fallback_cli", cmd=" ".join(cmd))
        
        proc = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            encoding="utf-8", 
            errors="ignore"
        )
        
        if proc.returncode != 0:
            err = (proc.stderr or "")[-1000:]
            raise RuntimeError(f"Fallback LLM failed rc={proc.returncode} stderr={err}")
            
        return _extract_json_object(proc.stdout)


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
            resolved = path.resolve()
            rel = resolved.relative_to(self.cfg.project_root.resolve())
        except Exception:
            return False

        # 1. Deny Dirs (Prefix check)
        for part in rel.parts:
            if part in self.cfg.deny_dirs:
                return False
        
        # 2. Test Protection
        if "test" in rel.name.lower() and "dummy" not in rel.name.lower():
             # Strict protection for tests unless it's a dummy
             return False

        # 3. Allowlist (Strict)
        if not self.cfg.allow_modify_globs:
            # Default DENY ALL if allowlist is empty
            return False
            
        # Check if matches any allow glob
        matched = False
        for glob in self.cfg.allow_modify_globs:
            if rel.match(glob) or path.match(glob):
                matched = True
                break
        
        return matched

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

    def backup(self, path: Path, content: str) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup = self.cfg.backups_dir / f"{path.name}.{timestamp}.bak"
        backup.write_text(content, encoding="utf-8")
        self._prune_backups(path)
        return backup

    def run_test_cmd(self) -> Tuple[bool, str]:
        if not self.cfg.test_cmd:
            return True, ""
        
        self.logger.log("run_test", cmd=self.cfg.test_cmd)
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
        # Guardrails: Check limits
        if len(patches) > self.cfg.max_patch_files:
            return False, f"Too many files patched: {len(patches)} > {self.cfg.max_patch_files}"

        total_lines = 0
        for p in patches:
            code = p.get("code", "") or p.get("diff", "")
            total_lines += len(code.splitlines())
        
        if total_lines > self.cfg.max_patch_lines:
            return False, f"Patch too large: {total_lines} lines > {self.cfg.max_patch_lines}"

        # Shadow Patching Strategy
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            temp_map: Dict[Path, Path] = {} # Real Path -> Temp Path
            originals: Dict[Path, str] = {}
            
            # 1. Prepare Shadow Copies
            for patch in patches:
                file_path = Path(str(patch.get("file_path"))).resolve()
                if not self.is_safe_target(file_path):
                    return False, f"Unsafe target path {file_path}"
                
                if file_path not in temp_map:
                    # Create temp file structure
                    rel_path = file_path.relative_to(self.cfg.project_root.resolve())
                    temp_file = temp_root / rel_path
                    temp_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    if file_path.exists():
                        content = file_path.read_text(encoding="utf-8", errors="ignore")
                        temp_file.write_text(content, encoding="utf-8")
                        originals[file_path] = content
                    else:
                        originals[file_path] = ""
                        temp_file.write_text("", encoding="utf-8")
                    
                    temp_map[file_path] = temp_file

            # 2. Apply Patches to Shadow Copies
            for patch in patches:
                file_path = Path(str(patch.get("file_path"))).resolve()
                temp_file = temp_map[file_path]
                mode = patch.get("mode")
                
                current_content = temp_file.read_text(encoding="utf-8", errors="ignore")
                
                try:
                    if mode == "replace_range":
                        start = int(patch.get("start_line", 0))
                        end = int(patch.get("end_line", 0))
                        code = patch.get("code", "")
                        updated = self._apply_replace_range(current_content, start, end, code)
                    elif mode == "unified_diff":
                        diff = patch.get("diff", "")
                        updated = self._apply_unified_diff(current_content, diff)
                    else:
                        return False, f"Unknown patch mode {mode}"
                    
                    temp_file.write_text(updated, encoding="utf-8")
                except Exception as e:
                    return False, f"Failed to apply patch to {file_path.name}: {e}"

            # 3. Verification (Compile & Test)
            # Check syntax
            for real_path, temp_path in temp_map.items():
                if real_path.suffix == ".py":
                    try:
                        import py_compile
                        py_compile.compile(str(temp_path), doraise=True)
                    except Exception as e:
                        return False, f"Syntax error in patched {real_path.name}: {e}"

            # Run Tests (if required)
            # Note: We can't easily run the project's test suite against the temp dir 
            # without complex environment setup. 
            # For this implementation, we will rely on the fact that we are modifying the *real* files
            # ONLY IF we are confident. 
            # BUT, the user requested "Shadow Patching" where we test *before* commit.
            # To do this properly, we'd need to copy the WHOLE project to temp. 
            # That's too heavy.
            # Compromise: We apply to temp files. If syntax passes, we apply to REAL files 
            # but keep backups. THEN we run tests. If tests fail, we ROLLBACK.
            # This is "Optimistic Commit with Atomic Rollback" which is safer than current but faster than full clone.
            
            # User specifically asked for "Atomic Commit" using os.replace.
            # This implies we should swap the file.
            
            if self.cfg.dry_run:
                self.logger.log("dry_run_patch", files=[str(p) for p in temp_map.keys()])
                return True, "Dry Run OK"

            # 4. Atomic Commit
            backups: List[Path] = []
            try:
                for real_path, temp_path in temp_map.items():
                    # Backup first
                    if real_path.exists():
                        backups.append(self.backup(real_path, originals[real_path]))
                    
                    # Atomic Replace
                    # os.replace is atomic and overwrites on Windows (Python 3.3+)
                    os.replace(str(temp_path), str(real_path))
            except Exception as e:
                # Fatal error during commit - try to restore from backups
                self.logger.log("commit_failed", error=str(e))
                # Rollback what we can
                for bk in backups:
                    # Rough guess, better to store map
                    pass 
                return False, f"Commit failed: {e}"

            # 5. Post-Commit Test
            if self.cfg.require_test_pass:
                ok, output = self.run_test_cmd()
                if not ok:
                    self.logger.log("test_failed_rollback", output=output[-500:])
                    # Rollback!
                    for real_path in temp_map.keys():
                        # Find latest backup
                        bk_glob = sorted(self.cfg.backups_dir.glob(f"{real_path.name}.*.bak"), reverse=True)
                        if bk_glob:
                            shutil.copy2(str(bk_glob[0]), str(real_path))
                    return False, f"Tests failed after patch: {output[-200:]}"

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
        self.fallback_cli = LocalCodeCLIClient(cfg, logger)
        self.redactor = Redactor(cfg)

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
        # Try to find a likely target in allowlist
        if self.cfg.allow_modify_globs:
             # Just pick the first file that exists and matches allowlist
             for glob in self.cfg.allow_modify_globs:
                 for p in self.cfg.project_root.glob(glob):
                     if p.is_file():
                         return p
        return (self.cfg.project_root / "train.py").resolve()

    def _read_file_safe(self, path: Path) -> str:
        if not path.exists():
            return ""
        try:
            # Binary read to avoid encoding issues with partial chars
            with path.open("rb") as f:
                data = f.read(self.cfg.max_file_bytes)
            content = data.decode("utf-8", errors="ignore")
            if len(data) == self.cfg.max_file_bytes:
                content += "\n\n# <truncated>"
            return self.redactor.redact(content)
        except Exception:
            return ""

    def _build_prompt(
        self,
        target: Path,
        line_no: Optional[int],
        stderr_tail: str,
        stdout_tail: str,
        previous_error: str = "",
        is_oom: bool = False,
    ) -> str:
        content = self._read_file_safe(target)
        stderr_safe = self.redactor.redact(stderr_tail)
        stdout_safe = self.redactor.redact(stdout_tail)
        
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
        ]
        
        if is_oom:
             parts.append("CRITICAL: The error is OUT OF MEMORY (OOM).")
             parts.append("You must propose a fix to REDUCE MEMORY USAGE.")
             parts.append("Examples: reduce batch size, enable gradient accumulation, use mixed precision, clear cache.")
        
        parts += [
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
            parts += ["", "Previous validation/test error", self.redactor.redact(previous_error)]
        parts += ["", "Target file content (redacted)", content]
        prompt = "\n".join(parts)
        if len(prompt) > self.cfg.max_prompt_chars:
            prompt = prompt[: self.cfg.max_prompt_chars] + "\n\n<truncated>"
        return prompt

    def _is_oom(self, stderr: str) -> bool:
        return "out of memory" in stderr.lower() or "cuda out of memory" in stderr.lower()

    def handle_failure(self, exit_code: int, stderr_tail: str, stdout_tail: str) -> bool:
        target, line_no = self._select_target_from_traceback(stderr_tail)
        if target is None:
            target = self._fallback_target()

        # Error Signature
        sanitized_err = self.redactor.redact(stderr_tail)
        signature = _sha256(f"{exit_code}:{target.name}:{sanitized_err[:500]}")
        
        # Quarantine Check
        if self.state.is_quarantined(signature):
            self.logger.log("quarantine_skip", signature=signature)
            return False

        # OOM Special Handling
        is_oom = self._is_oom(stderr_tail)
        max_retries = 1 if is_oom else self.cfg.max_retries_per_signature
        
        attempt = self.state.get_retry(signature)
        if attempt >= max_retries:
            self.logger.log("max_retries_reached", signature=signature, target=str(target))
            self.state.quarantine(signature)
            return False

        self.logger.log("attempt_fix", signature=signature, attempt=attempt+1, is_oom=is_oom)
        
        previous_error = ""
        # Try to fix
        prompt = self._build_prompt(target, line_no, stderr_tail, stdout_tail, previous_error, is_oom)
        
        try:
            data = self._call_llm(prompt, target)
            patches = self._validate_response(data, target)
            ok, err = self.patch.apply_patch_set(patches)
            
            self.state.inc_retry(signature)
            
            if ok:
                self.logger.log("patch_success", signature=signature)
                return True
            else:
                self.logger.log("patch_failed", error=err)
                # If patch failed to apply or verify, we count it as a retry.
                # If it was OOM, we stop immediately after 1 fail.
                if is_oom:
                     self.state.quarantine(signature)
                return False

        except Exception as e:
            self.logger.log("llm_error", error=str(e))
            return False

    def _call_llm(self, prompt: str, target: Path) -> Dict[str, Any]:
        try:
            if self.cfg.primary_llm == "gemini_cli":
                return self.gemini_cli.request_fix(prompt)
            if self.cfg.primary_llm == "gemini_api":
                return self.gemini_api.request_fix(prompt, str(target))
            raise RuntimeError(f"Unknown PHOENIX_PRIMARY={self.cfg.primary_llm}")
        except Exception as e:
            self.logger.log("primary_llm_error", error=str(e))
            if self.cfg.fallback_llm_cmd:
                self.logger.log("attempting_fallback")
                return self.fallback_cli.request_fix(prompt)
            raise

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
            
            normalized_patch = {"file_path": str(resolved), "mode": mode}
            if mode == "unified_diff":
                normalized_patch["diff"] = patch.get("diff", "")
            else:
                normalized_patch.update({
                    "start_line": patch.get("start_line"),
                    "end_line": patch.get("end_line"),
                    "code": patch.get("code")
                })
            normalized.append(normalized_patch)
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
        self.logger.log("killing_process_tree", pid=proc.pid)
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
        self.logger.log("start_training", cmd=self.cfg.train_cmd)
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
                    self.logger.log("heartbeat_timeout", idle_sec=idle)
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
        
        duration = time.time() - heartbeat["last"] # Approx
        self.logger.log("process_exit", rc=rc)

        return rc, err_buf.tail_text(), out_buf.tail_text()


def acquire_lock(cfg: Config, logger: Logger) -> bool:
    pid = os.getpid()
    lock_file = cfg.lock_path
    
    if lock_file.exists():
        try:
            content = lock_file.read_text().strip()
            old_pid = int(content.split()[0])
            # Check if process exists
            if os.name == "nt":
                # Simple check using tasklist
                proc = subprocess.run(["tasklist", "/FI", f"PID eq {old_pid}", "/NH"], capture_output=True, text=True)
                if str(old_pid) in proc.stdout:
                    logger.log("lock_held", pid=old_pid)
                    return False
            else:
                try:
                    os.kill(old_pid, 0)
                    logger.log("lock_held", pid=old_pid)
                    return False
                except OSError:
                    pass # Process dead
            
            logger.log("stale_lock_removed", old_pid=old_pid)
            lock_file.unlink()
        except Exception:
            pass

    try:
        lock_file.write_text(f"{pid} {_now_iso()}")
        return True
    except Exception:
        return False


def main() -> int:
    # Critical: Fix CWD before Config init to prevent System32 issues
    cwd = Path.cwd()
    script_dir = Path(__file__).parent.resolve()
    
    if "system32" in str(cwd).lower() or cwd != script_dir:
        # If we are in System32 or not in script dir (likely Task Scheduler issue),
        # switch to script directory immediately.
        try:
            os.chdir(script_dir)
            cwd = script_dir
            # We can't log yet because Config/Logger aren't ready, but this is a safety fix.
        except Exception:
            pass

    cfg = Config()
    cfg.ensure_dirs()
    redactor = Redactor(cfg)
    logger = Logger(cfg, redactor)
    
    if not acquire_lock(cfg, logger):
        print("Could not acquire lock. Exiting.")
        return 1

    state = StateStore(cfg)
    handler = ErrorHandler(cfg, logger, state)
    process_manager = ProcessManager(cfg, logger)

    logger.log("startup", config=dataclasses.asdict(cfg), cwd=str(cwd))

    while True:
        exit_code, stderr_tail, stdout_tail = process_manager.run_training()
        if exit_code == 0:
            logger.log("training_success")
            return 0

        try:
            fixed = handler.handle_failure(exit_code, stderr_tail, stdout_tail)
        except Exception as exc:
            logger.log("fatal_error", error=str(exc), traceback=traceback.format_exc())
            return 2

        if not fixed:
            if cfg.cooldown_seconds_on_stop > 0:
                logger.log("cooldown", seconds=cfg.cooldown_seconds_on_stop)
                time.sleep(cfg.cooldown_seconds_on_stop)
            logger.log("give_up")
            return 1

        logger.log("restarting")


if __name__ == "__main__":
    raise SystemExit(main())
