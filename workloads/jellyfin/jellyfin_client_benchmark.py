#!/usr/bin/env python3
"""
Client-side Jellyfin playback benchmark.

Outputs, per run:
  metadata.json
  timeseries.csv
  events.csv
  stalls.csv
  summary.csv
  raw.json
  final-frame.png

The script observes the HTML <video> element only. It does not require
server access and does not collect TCP congestion-control state.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

from playwright.async_api import Page, async_playwright


INSTRUMENTATION = r"""
(() => {
  if (window.__jfBench) return;

  const state = {
    createdPerfMs: performance.now(),
    createdEpochMs: performance.timeOrigin + performance.now(),
    measurementStartPerfMs: null,
    measurementStartEpochMs: null,
    firstPlayingPerfMs: null,
    firstPlayingEpochMs: null,
    firstPlayingMediaTimeS: null,
    events: [],
    samples: [],
    stalls: [],
    activeStall: null,
    video: null
  };

  const attached = new WeakSet();

  function perfMs() {
    return performance.now();
  }

  function epochMs() {
    return performance.timeOrigin + performance.now();
  }

  function finiteOrNull(value) {
    return Number.isFinite(value) ? value : null;
  }

  function bufferAheadS(video) {
    const t = video.currentTime;
    const ranges = video.buffered;

    for (let i = 0; i < ranges.length; i++) {
      const start = ranges.start(i);
      const end = ranges.end(i);
      if (t >= start - 0.05 && t <= end + 0.05) {
        return Math.max(0, end - t);
      }
    }
    return 0;
  }

  function frameStats(video) {
    if (typeof video.getVideoPlaybackQuality !== "function") {
      return {
        totalFrames: null,
        droppedFrames: null,
        corruptedFrames: null
      };
    }

    const q = video.getVideoPlaybackQuality();
    return {
      totalFrames: q.totalVideoFrames ?? null,
      droppedFrames: q.droppedVideoFrames ?? null,
      corruptedFrames: q.corruptedVideoFrames ?? null
    };
  }

  function currentRecord(type, video) {
    return {
      type,
      perf_ms: perfMs(),
      epoch_ms: epochMs(),
      media_time_s: video.currentTime,
      duration_s: finiteOrNull(video.duration),
      buffer_ahead_s: bufferAheadS(video),
      paused: video.paused,
      seeking: video.seeking,
      ended: video.ended,
      ready_state: video.readyState,
      network_state: video.networkState,
      playback_rate: video.playbackRate
    };
  }

  function finishActiveStall(video, endPerfMs, endEpochMs) {
    if (!state.activeStall) return;

    state.stalls.push({
      start_perf_ms: state.activeStall.startPerfMs,
      start_epoch_ms: state.activeStall.startEpochMs,
      end_perf_ms: endPerfMs,
      end_epoch_ms: endEpochMs,
      duration_ms: endPerfMs - state.activeStall.startPerfMs,
      start_media_time_s: state.activeStall.startMediaTimeS,
      end_media_time_s: video.currentTime,
      buffer_before_s: state.activeStall.bufferBeforeS,
      buffer_after_s: bufferAheadS(video),
      incomplete: false
    });
    state.activeStall = null;
  }

  function attach(video) {
    if (attached.has(video)) return;
    attached.add(video);
    state.video = video;

    const eventTypes = [
      "loadstart", "loadedmetadata", "loadeddata", "canplay",
      "play", "playing", "waiting", "stalled", "pause",
      "seeking", "seeked", "ended", "emptied", "error"
    ];

    for (const eventType of eventTypes) {
      video.addEventListener(eventType, () => {
        const record = currentRecord(eventType, video);
        state.events.push(record);

        if (eventType === "play" && state.measurementStartPerfMs === null) {
          state.measurementStartPerfMs = record.perf_ms;
          state.measurementStartEpochMs = record.epoch_ms;
        }

        if (eventType === "playing") {
          if (state.firstPlayingPerfMs === null) {
            state.firstPlayingPerfMs = record.perf_ms;
            state.firstPlayingEpochMs = record.epoch_ms;
            state.firstPlayingMediaTimeS = video.currentTime;
          }
          finishActiveStall(video, record.perf_ms, record.epoch_ms);
        }

        if (
          eventType === "waiting" &&
          state.firstPlayingPerfMs !== null &&
          !video.paused &&
          !video.seeking &&
          state.activeStall === null
        ) {
          state.activeStall = {
            startPerfMs: record.perf_ms,
            startEpochMs: record.epoch_ms,
            startMediaTimeS: video.currentTime,
            bufferBeforeS: bufferAheadS(video)
          };
        }

        // A deliberate seek is not a network rebuffer.
        if (eventType === "seeking") {
          state.activeStall = null;
        }
      });
    }
  }

  function discoverVideo() {
    for (const video of document.querySelectorAll("video")) {
      attach(video);
    }
  }

  new MutationObserver(discoverVideo).observe(document, {
    childList: true,
    subtree: true
  });

  discoverVideo();

  const timer = setInterval(() => {
    const video = state.video;
    if (!video) return;

    // Jellyfin can create and start the video before our event listener
    // observes the initial "playing" event. Detect that state directly.
    if (
      state.firstPlayingPerfMs === null &&
      !video.paused &&
      !video.ended &&
      video.readyState >= 2
    ) {
      const detectedPerfMs = perfMs();
      const detectedEpochMs = epochMs();

      if (state.measurementStartPerfMs === null) {
        state.measurementStartPerfMs = detectedPerfMs;
        state.measurementStartEpochMs = detectedEpochMs;
      }

      state.firstPlayingPerfMs = detectedPerfMs;
      state.firstPlayingEpochMs = detectedEpochMs;
      state.firstPlayingMediaTimeS = video.currentTime;

      state.events.push(
        currentRecord("playing-detected-by-poll", video)
      );
    }

    const frames = frameStats(video);
    state.samples.push({
      perf_ms: perfMs(),
      epoch_ms: epochMs(),
      media_time_s: video.currentTime,
      duration_s: finiteOrNull(video.duration),
      buffer_ahead_s: bufferAheadS(video),
      paused: video.paused,
      seeking: video.seeking,
      ended: video.ended,
      ready_state: video.readyState,
      network_state: video.networkState,
      playback_rate: video.playbackRate,
      total_frames: frames.totalFrames,
      dropped_frames: frames.droppedFrames,
      corrupted_frames: frames.corruptedFrames
    });
  }, 250);

  window.__jfBench = {
    snapshot() {
      const copy = {
        createdPerfMs: state.createdPerfMs,
        createdEpochMs: state.createdEpochMs,
        measurementStartPerfMs: state.measurementStartPerfMs,
        measurementStartEpochMs: state.measurementStartEpochMs,
        firstPlayingPerfMs: state.firstPlayingPerfMs,
        firstPlayingEpochMs: state.firstPlayingEpochMs,
        firstPlayingMediaTimeS: state.firstPlayingMediaTimeS,
        events: [...state.events],
        samples: [...state.samples],
        stalls: [...state.stalls]
      };

      if (state.activeStall && state.video) {
        const endPerfMs = perfMs();
        copy.stalls.push({
          start_perf_ms: state.activeStall.startPerfMs,
          start_epoch_ms: state.activeStall.startEpochMs,
          end_perf_ms: endPerfMs,
          end_epoch_ms: epochMs(),
          duration_ms: endPerfMs - state.activeStall.startPerfMs,
          start_media_time_s: state.activeStall.startMediaTimeS,
          end_media_time_s: state.video.currentTime,
          buffer_before_s: state.activeStall.bufferBeforeS,
          buffer_after_s: bufferAheadS(state.video),
          incomplete: true
        });
      }
      return copy;
    },

    stop() {
      clearInterval(timer);
    }
  };
})();
"""


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    fraction = position - lo
    return ordered[lo] * (1 - fraction) + ordered[hi] * fraction


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def flatten_summary(
    snapshot: dict[str, Any],
    run_id: str,
    label: str,
    target_media_s: float,
) -> dict[str, Any]:
    start_ms = snapshot.get("measurementStartPerfMs")
    first_playing_ms = snapshot.get("firstPlayingPerfMs")

    startup_delay_ms = None
    if start_ms is not None and first_playing_ms is not None:
        startup_delay_ms = max(0.0, first_playing_ms - start_ms)

    post_start_samples = [
        sample
        for sample in snapshot["samples"]
        if first_playing_ms is not None and sample["perf_ms"] >= first_playing_ms
    ]

    buffers = [
        float(sample["buffer_ahead_s"])
        for sample in post_start_samples
        if sample.get("buffer_ahead_s") is not None
    ]

    stalls = snapshot["stalls"]
    stall_durations = [float(stall["duration_ms"]) for stall in stalls]
    total_stall_ms = sum(stall_durations)

    first_media = snapshot.get("firstPlayingMediaTimeS")
    final_media = post_start_samples[-1]["media_time_s"] if post_start_samples else None

    played_media_s = None
    if first_media is not None and final_media is not None:
        played_media_s = max(0.0, float(final_media) - float(first_media))

    time_to_first_stall_s = None
    if stalls and first_playing_ms is not None:
        time_to_first_stall_s = max(
            0.0, (float(stalls[0]["start_perf_ms"]) - first_playing_ms) / 1000.0
        )

    frame_samples = [
        s for s in post_start_samples if s.get("total_frames") is not None
    ]
    total_frames = None
    dropped_frames = None
    dropped_frame_ratio = None
    if len(frame_samples) >= 2:
        total_frames = (
            int(frame_samples[-1]["total_frames"])
            - int(frame_samples[0]["total_frames"])
        )
        dropped_frames = (
            int(frame_samples[-1]["dropped_frames"])
            - int(frame_samples[0]["dropped_frames"])
        )
        if total_frames > 0:
            dropped_frame_ratio = dropped_frames / total_frames

    return {
        "run_id": run_id,
        "label": label,
        "target_media_s": target_media_s,
        "startup_delay_ms": startup_delay_ms,
        "stall_count": len(stalls),
        "total_stall_ms": total_stall_ms,
        "mean_stall_ms": statistics.mean(stall_durations) if stall_durations else 0.0,
        "max_stall_ms": max(stall_durations, default=0.0),
        "rebuffer_ratio": total_stall_ms / (target_media_s * 1000.0),
        "stall_free": len(stalls) == 0,
        "time_to_first_stall_s": time_to_first_stall_s,
        "played_media_s": played_media_s,
        "completion_ratio": (
            min(1.0, played_media_s / target_media_s)
            if played_media_s is not None
            else None
        ),
        "mean_buffer_s": statistics.mean(buffers) if buffers else None,
        "median_buffer_s": statistics.median(buffers) if buffers else None,
        "p05_buffer_s": percentile(buffers, 0.05),
        "minimum_buffer_s": min(buffers) if buffers else None,
        "fraction_buffer_below_1s": (
            sum(v < 1.0 for v in buffers) / len(buffers) if buffers else None
        ),
        "fraction_buffer_below_2s": (
            sum(v < 2.0 for v in buffers) / len(buffers) if buffers else None
        ),
        "fraction_buffer_below_5s": (
            sum(v < 5.0 for v in buffers) / len(buffers) if buffers else None
        ),
        "total_frames": total_frames,
        "dropped_frames": dropped_frames,
        "dropped_frame_ratio": dropped_frame_ratio,
    }


async def click_play(page: Page) -> bool:
    """
    Click only Jellyfin's main Play or Resume button.

    We deliberately require the visible text or aria-label to be exactly
    'Play' or 'Resume' so that controls such as 'Play in group' are ignored.
    """
    deadline = time.monotonic() + 60.0

    while time.monotonic() < deadline:
        result = await page.evaluate(
            """
            () => {
                function isVisible(element) {
                    const style = window.getComputedStyle(element);
                    const rect = element.getBoundingClientRect();

                    return (
                        style.display !== "none" &&
                        style.visibility !== "hidden" &&
                        Number(style.opacity) !== 0 &&
                        rect.width > 0 &&
                        rect.height > 0
                    );
                }

                function normalize(value) {
                    return (value || "")
                        .replace(/\\s+/g, " ")
                        .trim()
                        .toLowerCase();
                }

                const controls = Array.from(
                    document.querySelectorAll("button, [role='button']")
                );

                const candidates = controls.filter((element) => {
                    if (!isVisible(element)) {
                        return false;
                    }

                    const text = normalize(element.innerText);
                    const ariaLabel = normalize(
                        element.getAttribute("aria-label")
                    );
                    const title = normalize(element.getAttribute("title"));

                    return (
                        text === "play" ||
                        text === "resume" ||
                        ariaLabel === "play" ||
                        ariaLabel === "resume" ||
                        title === "play" ||
                        title === "resume"
                    );
                });

                if (candidates.length === 0) {
                    return null;
                }

                // Prefer a button with exact visible text.
                const button =
                    candidates.find((element) => {
                        const text = normalize(element.innerText);
                        return text === "play" || text === "resume";
                    }) || candidates[0];

                button.scrollIntoView({
                    block: "center",
                    inline: "center"
                });

                button.click();

                return {
                    text: normalize(button.innerText),
                    ariaLabel: button.getAttribute("aria-label"),
                    title: button.getAttribute("title"),
                    tag: button.tagName
                };
            }
            """
        )

        if result is not None:
            print(f"Clicked main playback button: {result}")
            return True

        await page.wait_for_timeout(500)

    print("Could not find an exact Play or Resume button.")
    return False


async def wait_for_video_start(page: Page, timeout_s: float) -> None:
    """
    Wait until either the instrumentation observed playback or the actual
    HTML video element is visibly in a playable, non-paused state.
    """
    await page.wait_for_function(
        """
        () => {
            const snapshot = window.__jfBench?.snapshot?.();

            if (snapshot?.firstPlayingPerfMs !== null &&
                snapshot?.firstPlayingPerfMs !== undefined) {
                return true;
            }

            const video = document.querySelector("video");

            return Boolean(
                video &&
                !video.paused &&
                !video.ended &&
                video.readyState >= 2
            );
        }
        """,
        timeout=int(timeout_s * 1000),
        polling=250,
    )

    # Give the 250 ms instrumentation sampler time to record the state.
    await page.wait_for_timeout(500)


async def run(args: argparse.Namespace) -> None:
    run_id = args.run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    run_dir = Path(args.output) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "run_id": run_id,
        "label": args.label,
        "url": args.url,
        "target_media_s": args.duration,
        "sample_interval_ms": 250,
        "browser": args.browser,
        "headless": args.headless,
        "created_epoch_ns": time.time_ns(),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    async with async_playwright() as p:
        browser_type = getattr(p, args.browser)
        browser = await browser_type.launch(
            headless=args.headless,
            args=["--autoplay-policy=no-user-gesture-required"]
            if args.browser == "chromium"
            else None,
        )

        context_kwargs: dict[str, Any] = {
            "viewport": {"width": 1600, "height": 900},
        }
        auth_path = Path(args.auth)
        if auth_path.exists():
            context_kwargs["storage_state"] = str(auth_path)

        context = await browser.new_context(**context_kwargs)
        await context.add_init_script(script=INSTRUMENTATION)
        page = await context.new_page()

        page.on("console", lambda msg: print(f"[browser:{msg.type}] {msg.text}"))
        page.on("pageerror", lambda err: print(f"[page error] {err}"))

        print(f"Opening: {args.url}")
        await page.goto(args.url, wait_until="domcontentloaded", timeout=60_000)

        # Verify that the benchmark collector was installed. The context
        # init script can execute before Jellyfin's document is fully built,
        # so explicitly install it again on the loaded document if needed.
        collector_ready = await page.evaluate(
            "() => Boolean(window.__jfBench?.snapshot)"
        )

        if not collector_ready:
            print("Collector missing after navigation; installing it directly.")
            await page.evaluate(INSTRUMENTATION)

            collector_ready = await page.evaluate(
                "() => Boolean(window.__jfBench?.snapshot)"
            )

        if not collector_ready:
            raise RuntimeError(
                "Could not install the Jellyfin browser collector."
            )

        print("Benchmark collector installed.")

        if args.manual_play:
            print("Click Play in the opened browser.")
        else:
            clicked = await click_play(page)
            if clicked:
                print("Clicked Play automatically.")
            else:
                print("Could not find Play automatically; click it in the browser.")

        try:
            await wait_for_video_start(page, args.start_timeout)
        except Exception as exc:
            await page.screenshot(path=str(run_dir / "start-failure.png"))
            raise RuntimeError(
                "Playback did not start. Check login, item access, and the Play button. "
                f"A screenshot was saved to {run_dir / 'start-failure.png'}"
            ) from exc

        await page.wait_for_selector("video", state="attached", timeout=30_000)
        start_media_s = await page.locator("video").first.evaluate(
            "(video) => video.currentTime"
        )

        print(f"Playback started. Measuring {args.duration:.1f} media seconds...")
        wall_deadline = time.monotonic() + args.duration + args.timeout_margin
        last_snapshot = None

        while time.monotonic() < wall_deadline:
            state = await page.locator("video").first.evaluate(
                """video => ({
                    currentTime: video.currentTime,
                    ended: video.ended,
                    error: video.error ? video.error.code : null
                })"""
            )

            if state["error"] is not None:
                raise RuntimeError(f"HTML video error code: {state['error']}")

            if float(state["currentTime"]) - float(start_media_s) >= args.duration:
                break

            if state["ended"]:
                break

            # Preserve data continuously. Jellyfin may recreate the page's
            # JavaScript context near the end of playback.
            try:
                candidate = await page.evaluate(
                    "() => window.__jfBench?.snapshot?.() ?? null"
                )

                if candidate is not None:
                    candidate_samples = len(candidate.get("samples", []))
                    saved_samples = (
                        len(last_snapshot.get("samples", []))
                        if last_snapshot is not None
                        else -1
                    )

                    # Keep the most complete snapshot seen so far.
                    if candidate_samples >= saved_samples:
                        last_snapshot = candidate

            except Exception:
                # A transient navigation can destroy the execution context.
                # The previous valid snapshot remains available.
                pass

            await asyncio.sleep(0.25)
        else:
            print("Warning: wall-clock timeout reached before target media duration.")

        # Try one final snapshot, but fall back to the continuously saved one
        # if Jellyfin has recreated the JavaScript context.
        try:
            candidate = await page.evaluate(
                "() => window.__jfBench?.snapshot?.() ?? null"
            )

            if candidate is not None:
                candidate_samples = len(candidate.get("samples", []))
                saved_samples = (
                    len(last_snapshot.get("samples", []))
                    if last_snapshot is not None
                    else -1
                )

                if candidate_samples >= saved_samples:
                    last_snapshot = candidate

        except Exception:
            pass

        if last_snapshot is None:
            raise RuntimeError(
                "Playback occurred, but no benchmark snapshot was captured."
            )

        snapshot = last_snapshot

        try:
            await page.evaluate(
                "() => window.__jfBench?.stop?.()"
            )
        except Exception:
            pass
        await page.screenshot(path=str(run_dir / "final-frame.png"))

        try:
            await page.locator("video").first.evaluate("(video) => video.pause()")
        except Exception:
            pass

        await context.close()
        await browser.close()

    summary = flatten_summary(
        snapshot=snapshot,
        run_id=run_id,
        label=args.label,
        target_media_s=args.duration,
    )

    (run_dir / "raw.json").write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8"
    )
    write_csv(run_dir / "timeseries.csv", snapshot["samples"])
    write_csv(run_dir / "events.csv", snapshot["events"])

    stalls_with_index = [
        {"stall_index": index + 1, **stall}
        for index, stall in enumerate(snapshot["stalls"])
    ]
    write_csv(run_dir / "stalls.csv", stalls_with_index)
    write_csv(run_dir / "summary.csv", [summary])

    print(json.dumps(summary, indent=2))
    print(f"Results: {run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Client-side Jellyfin QoE benchmark"
    )
    parser.add_argument("--url", required=True, help="Jellyfin item details URL")
    parser.add_argument("--auth", default="auth.json", help="Playwright storage state")
    parser.add_argument("--output", default="results")
    parser.add_argument("--run-id")
    parser.add_argument("--label", default="unlabelled")
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--start-timeout", type=float, default=120.0)
    parser.add_argument("--timeout-margin", type=float, default=300.0)
    parser.add_argument(
        "--browser",
        choices=("chromium", "firefox", "webkit"),
        default="chromium",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--manual-play", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
