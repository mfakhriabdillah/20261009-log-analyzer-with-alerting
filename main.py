```python
#!/usr/bin/env python3
"""
Log Analyzer with Alerting — CLI tool for cloud/DevOps engineers.
Analyzes log files for patterns, errors, and anomalies, with configurable alerting.
"""

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_logging(log_level: str, verbose: bool) -> None:
    """Configure root logger with appropriate level and format."""
    level = logging.DEBUG if verbose else getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def parse_timestamp(line: str) -> Optional[datetime]:
    """Extract a datetime from common log timestamp formats.

    Supported patterns:
      - ISO-8601: 2024-03-15T10:23:45Z or 2024-03-15T10:23:45+00:00
      - Syslog:   Mar 15 10:23:45
      - Apache:   [15/Mar/2024:10:23:45 +0000]
    Returns None when no pattern matches.
    """
    patterns = [
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)",
        r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})",
        r"\[(\d{2}/[A-Z][a-z]{2}/\d{4}:\d{2}:\d{2}:\d{2}\s*[+-]\d{4})\]",
    ]
    for pat in patterns:
        m = re.search(pat, line)
        if m:
            ts_str = m.group(1)
            for fmt in (
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%d/%b/%Y:%H:%M:%S %z",
            ):
                try:
                    return datetime.strptime(ts_str, fmt)
                except ValueError:
                    continue
            # Fallback: assume current year for syslog-style timestamps
            try:
                now = datetime.now()
                parsed = datetime.strptime(ts_str, "%b %d %H:%M:%S")
                return parsed.replace(year=now.year)
            except ValueError:
                pass
    return None


# ---------------------------------------------------------------------------
# Core classes
# ---------------------------------------------------------------------------

class AlertRule:
    """Defines a single alert condition."""

    def __init__(
        self,
        name: str,
        pattern: str,
        threshold: int = 1,
        window_seconds: float = 60.0,
        severity: str = "warning",
    ):
        self.name = name
        self.compiled = re.compile(pattern, re.IGNORECASE)
        self.threshold = threshold
        self.window = window_seconds
        self.severity = severity.lower()
        self._events: list[float] = []

    def evaluate(self, line: str, ts: Optional[datetime]) -> bool:
        """Return True when the rule fires for the given line."""
        if not self.compiled.search(line):
            return False
        now = time.time()
        self._events.append(now)
        cutoff = now - self.window
        self._events = [t for t in self._events if t > cutoff]
        return len(self._events) >= self.threshold

    def reset(self) -> None:
        self._events.clear()


class LogAnalyzer:
    """Analyzes one or more log files and produces summary + alerts."""

    SEVERITY_LEVELS = {
        "error": 0,
        "err": 0,
        "fatal": 0,
        "critical": 0,
        "crit": 0,
        "warn": 1,
        "warning": 1,
        "info": 2,
        "notice": 2,
        "debug": 3,
        "trace": 3,
    }

    def __init__(
        self,
        patterns: list[str],
        alert_rules: list[AlertRule],
        ignore_patterns: list[str],
        top_n: int = 10,
    ):
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
        self.alert_rules = alert_rules
        self.compiled_ignores = [
            re.compile(p, re.IGNORECASE) for p in ignore_patterns
        ]
        self.top_n = top_n
        self.line_count = 0
        self.match_count = 0
        self.ignored_count = 0
        self.severity_counter: Counter = Counter()
        self.error_messages: Counter = Counter()
        self.alerts_triggered: list[dict] = []
        self.timestamps: list[datetime] = []

    def _should_ignore(self, line: str) -> bool:
        return any(r.search(line) for r in self.compiled_ignores)

    def _classify_severity(self, line: str) -> Optional[str]:
        lower = line.lower()
        for keyword, rank in self.SEVERITY_LEVELS.items():
            if keyword in lower:
                return keyword
        return None

    def analyze(self, filepath: Path) -> None:
        """Process a single log file."""
        logger = logging.getLogger(__name__)
        logger.info("Analyzing %s", filepath)

        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                line = raw_line.rstrip("\n\r")
                if not line:
                    continue
                self.line_count += 1

                if self._should_ignore(line):
                    self.ignored_count += 1
                    continue

                matched = False
                for pattern in self.compiled_patterns:
                    if pattern.search(line):
                        matched = True
                        self.match_count += 1
                        sev = self._classify_severity(line)
                        if sev:
                            self.severity_counter[sev] += 1
                        # Track error/exception messages for top-N reporting
                        if sev in ("error", "fatal", "critical", "warn"):
                            # Strip timestamp prefix for cleaner message grouping
                            clean = re.sub(
                                r"^[\dT:\-+Z\s.\[\]]+", "", line
                            ).strip()
                            self.error_messages[clean] += 1
                        break  # one match per line is enough

                ts = parse_timestamp(line)
                if ts:
                    self.timestamps.append(ts)

                for rule in self.alert_rules:
                    if rule.evaluate(line, ts):
                        alert = {
                            "rule": rule.name,
                            "severity": rule.severity,
                            "timestamp": datetime.now().isoformat(),
                            "line": line[:500],
                            "context_window": f"{rule.window}s / threshold={rule.threshold}",
                        }
                        self.alerts_triggered.append(alert)
                        rule.reset()
                        logger.warning(
                            "ALERT [%s] %s — %s",
                            rule.severity.upper(),
                            rule.name,
                            line[:200],
                        )

    def report(self) -> dict:
        """Build a summary report as a dictionary."""
        span: Optional[str] = None
        if len(self.timestamps) >= 2:
            ordered = sorted(self.timestamps)
            delta = ordered[-1] - ordered[0]
            span = str(timedelta(seconds=int(delta.total_seconds())))

        return {
            "file": "multiple",
            "lines_processed": self.line_count,
            "lines_matched": self.match_count,
            "lines_ignored": self.ignored_count,
            "time_span": span,
            "first_seen": (
                sorted(self.timestamps)[0].isoformat()
                if self.timestamps
                else None
            ),
            "last_seen": (
                sorted(self.timestamps)[-1].isoformat()
                if self.timestamps
                else None
            ),
            "severity_counts": dict(self.severity_counter.most_common()),
            "top_error_messages": self.error_messages.most_common(self.top_n),
            "alerts_triggered": self.alerts_triggered,
            "alert_summary": {
                a["severity"]: sum(
                    1 for x in self.alerts_triggered if x["severity"] == a["severity"]
                )
                for a in {x["severity"] for x in self.alerts_triggered}
            },
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_alert_rules(args: argparse.Namespace) -> list[AlertRule]:
    """Construct AlertRule instances from CLI flags."""
    rules: list[AlertRule] = []
    flags = [
        ("error_spike", "error|fatal|critical", "err_spike"),
        ("auth_failure", "authentication fail|login fail|401|unauthorized", "auth_fail"),
        ("oom", "out of memory|oom|killed process", "oom_kill"),
        ("timeout", "timed? ?out|connection reset|504", "timeout"),
        ("disk_warn", "disk|space|no space left|inode", "disk_full"),
    ]
    for flag_name, pattern, default_severity in flags:
        if getattr(args, flag_name, False):
            rules.append(
                AlertRule(
                    name=flag_name,
                    pattern=pattern,
                    threshold=args.threshold,
                    window_seconds=args.window,
                    severity=default_severity,
                )
            )
    # Custom rules
    for custom in (args.alert_custom or []):
        parts = custom.split(":", 1)
        if len(parts) != 2:
            logging.getLogger(__name__).error("Bad custom alert rule: %s", custom)
            continue
        name, pattern = parts
        rules.append(
            AlertRule(
                name=name,
                pattern=pattern,
                threshold=args.threshold,
                window_seconds=args.window,
                severity="warning",
            )
        )
    return rules


def validate_files(paths: list[str]) -> list[Path]:
    """Return resolved, existing file paths or exit with an error."""
    out: list[Path] = []
    for p in paths:
        path = Path(p).expanduser()
        if not path.exists():
            logging.error("File not found: %s", path)
            sys.exit(1)
        if not path.is_file():
            logging.error("Not a regular file: %s", path)
            sys.exit(1)
        out.append(path)
    return out


def cli_arguments() -> argparse.Namespace:
    """Parse and return CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="loganalyzer",
        description="Analyze log files for patterns and trigger alerts (cloud/DevOps).",
        epilog="Example: loganalyzer --pattern 'Exception' --error-spike --output report.json app.log",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "logs",
        nargs="+",
        help="One or more log file paths to analyze.",
    )
    parser.add_argument(
        "--pattern", "-p",
        action="append",
        default=[],
        metavar="REGEX",
        help="Regex pattern(s) to match. Can be repeated. All are ORed together.",
    )
    parser.add_argument(
        "--ignore", "-i",
        action="append",
        default=[],
        metavar="REGEX",
        help="Regex pattern(s) to ignore/skip. Can be repeated.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        metavar="N",
        help="Number of top error messages to surface (default: 10).",
    )

    # Alerting controls
    alerting = parser.add_argument_group("alerting options")
    alerting.add_argument(
        "--error-spike",
        action="store_true",
        help="Alert on repeated error/fatal/critical lines.",
    )
    alerting.add_argument(
        "--auth-failure",
        action="store_true",
        help="Alert on authentication/login failures.",
    )
    alerting.add_argument(
        "--oom",
        action="store_true",
        help="Alert on out-of-memory kills.",
    )
    alerting.add_argument(
        "--timeout",
        action="store_true",
        help="Alert on timeout/connection-reset events.",
    )
    alerting.add_argument(
        "--disk-warn",
        action="store_true",
        help="Alert on disk-space warnings.",
    )
    alerting.add_argument(
        "--alert-custom",
        action="append",
        metavar="NAME:PATTERN",
        help="Custom alert rule as NAME:PATTERN. Can be repeated.",
    )
    alerting.add_argument(
        "--threshold",
        type=int,
        default=3,
        metavar="N",
        help="Consecutive matches within the window before alert fires (default: 3).",
    )
    alerting.add_argument(
        "--window",
        type=float,
        default=60.0,
        metavar="SECS",
        help="Time window in seconds for alert thresholds (default: 60).",
    )

    # Output
    output = parser.add_argument_group("output options")
    output.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Write JSON report to this file instead of stdout.",
    )
    output.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress console output; only write report file.",
    )

    # Logging controls
    loggrp = parser.add_argument_group("logging options")
    loggrp.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )
    loggrp.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    return parser.parse_args()


def main() -> int:
    """Entry point. Returns exit code."""
    args = cli_arguments()
    setup_logging(args.log_level, args.verbose)
    logger = logging.getLogger(__name__)

    try:
        files = validate_files(args.logs)
    except Exception as exc:
        logger.error("Validation failed: %s", exc)
        return 1

    alert_rules = build_alert_rules(args)

    analyzer = LogAnalyzer(
        patterns=args.pattern,
        alert_rules=alert_rules,
        ignore_patterns=args.ignore,
        top_n=args.top_n,
    )

    for filepath in files:
        try:
            analyzer.analyze(filepath)
        except OSError as exc:
            logger.error("Failed to read %s: %s", filepath, exc)
            return 1

    report = analyzer.report()

    # Print summary to stdout unless --quiet
    if not args.quiet:
        print("\n=== Log Analysis Report ===")
        print(f"Lines processed : {report['lines_processed']}")
        print(f"Lines matched   : {report['lines_matched']}")
        print(f"Lines ignored   : {report['lines_ignored']}")
        if report["time_span"]:
            print(f"Time span       : {report['time_span']}")
        if report["first_seen"]:
            print(f"First seen      : {report['first_seen']}")
        if report["last_seen"]:
            print(f"Last seen       : {report['last_seen']}")
        if report["severity_counts"]:
            print("\nSeverity counts:")
            for sev, count in report["severity_counts"].items():
                print(f"  {sev:12s}  {count}")
        if report["top_error_messages"]:
            print(f"\nTop error messages (showing <= {args.top_n}):")
            for msg, count in report["top_error_messages"]:
                snippet = msg[:120].replace("\n", " ")
                print(f"  [{count:>4}] {snippet}")
        if report["alerts_triggered"]:
            print(f"\n⚠  Alerts triggered: {len(report['alerts_triggered'])}")
            for alert in report["alerts_triggered"]:
                print(
                    f"  [{alert['severity'].upper():>8}] {alert['rule']}"
                    f" @ {alert['timestamp']}"
                )
        else:
            print("\nNo alerts triggered.")
        print("===========================\n")

    # Write JSON report
    json_report = json.dumps(report, indent=2, default=str)
    if args.output:
        try:
            out_path = Path(args.output).expanduser()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json_report, encoding="utf-8")
            logger.info("Report written to %s", out_path)
        except OSError as exc:
            logger.error("Failed to write report: %s", exc)
            return 1
    elif not args.quiet:
        print(json_report)

    # Non-zero exit if any alerts fired
    return 2 if report["alerts_triggered"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
```