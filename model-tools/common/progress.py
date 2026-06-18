"""统一的日志 / 进度工具。

使用 ``rich`` 输出彩色日志；如果 rich 不可用则回退到普通 print。
所有函数都是**幂等**且**线程安全**（基于 rich.console）。
"""

from __future__ import annotations

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    _console = Console(stderr=True)

    def _print(kind: str, msg: str) -> None:
        style_map = {
            "step":  ("🔹", "bold cyan"),
            "info":  ("ℹ️ ", "blue"),
            "ok":    ("✅", "bold green"),
            "warn":  ("⚠️ ", "yellow"),
            "fail":  ("❌", "bold red"),
        }
        emoji, style = style_map.get(kind, ("", "white"))
        _console.print(f"{emoji} {msg}", style=style)

    def step(msg: str) -> None:  # noqa: D401
        """打印一个步骤标题，例如 ``[1/5] 加载模型``。"""
        _print("step", msg)

    def info(msg: str) -> None:
        _print("info", msg)

    def ok(msg: str) -> None:
        _print("ok", msg)

    def warn(msg: str) -> None:
        _print("warn", msg)

    def fail(msg: str) -> None:
        _print("fail", msg)

    def make_progress() -> Progress:
        """构造一个下载/解压进度条。"""
        return Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            DownloadColumn(),
            "•",
            TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(),
            transient=True,
            console=_console,
        )

except ImportError:  # pragma: no cover - rich 不可用时的回退
    def step(msg: str) -> None:  # type: ignore[no-redef]
        print(f"[STEP] {msg}")

    def info(msg: str) -> None:  # type: ignore[no-redef]
        print(f"[INFO] {msg}")

    def ok(msg: str) -> None:  # type: ignore[no-redef]
        print(f"[OK]   {msg}")

    def warn(msg: str) -> None:  # type: ignore[no-redef]
        print(f"[WARN] {msg}")

    def fail(msg: str) -> None:  # type: ignore[no-redef]
        print(f"[FAIL] {msg}", flush=True)

    def make_progress():  # type: ignore[no-redef]
        return None
