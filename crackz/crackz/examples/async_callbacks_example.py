"""Example demonstrating async callback patterns in Crackz.

This module provides examples of how to use the new async callback infrastructure
alongside existing synchronous callbacks, showing backward compatibility and the
benefits of the queue-based approach.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crackz import Project


def example_sync_callbacks(project: Project) -> None:
    """Example of traditional synchronous callback usage (backward compatible).

    This is the existing pattern that continues to work without changes.

    Args:
        project: Project configuration
    """
    from crackz.ai.training import train

    # Traditional sync callbacks - still supported
    def progress_callback(progress: float, message: str | None = None) -> None:
        """Synchronous progress callback."""
        print(f"Progress: {progress:.2%} - {message}")

    def metrics_callback(metrics: dict[str, float]) -> None:
        """Synchronous metrics callback."""
        print(f"Metrics: {metrics}")

    # Works exactly as before - no changes needed
    train(
        project,
        progress_reporter=progress_callback,
        metrics_callback=metrics_callback,
    )


def example_async_callbacks_with_dispatcher(project: Project) -> None:
    """Example using async callback dispatcher for non-blocking updates.

    This shows how to use the new queue-based async pattern for better
    performance when dealing with high-frequency callbacks.

    Args:
        project: Project configuration
    """
    from crackz.ai.core.async_callbacks import CallbackDispatcher
    from crackz.ai.training import train

    # Create dispatcher for queue-based communication
    dispatcher = CallbackDispatcher(maxsize=100)

    # Create sync reporters that feed async queues
    progress_reporter = dispatcher.create_sync_progress_reporter()
    metrics_reporter = dispatcher.create_sync_metrics_reporter()

    # Pass to training - works just like sync callbacks
    # But now updates go through queues for non-blocking behavior
    train(
        project,
        progress_reporter=progress_reporter,
        metrics_callback=metrics_reporter,
    )


def example_async_consumers_in_gui(project: Project) -> None:
    """Example of using async consumers in GUI for real-time updates.

    This demonstrates the full async pattern with consumers running in
    a separate event loop, ideal for GUI applications.

    Note: This is a simplified example. For production use, prefer the
    `example_async_with_gui_helpers` pattern which properly manages
    event loop lifecycle and cleanup.

    Args:
        project: Project configuration
    """
    import asyncio
    import threading
    import time

    from crackz.ai.core.async_callbacks import (
        CallbackDispatcher,
        create_async_metrics_consumer,
        create_async_progress_consumer,
    )
    from crackz.ai.training import train

    # Create dispatcher
    dispatcher = CallbackDispatcher(maxsize=100)

    # Use threading.Event for clean synchronization
    loop_ready = threading.Event()
    event_loop_holder = {"loop": None}

    # Define GUI update callbacks (these run in main thread)
    def update_progress_ui(progress: float, message: str | None = None) -> None:
        """Update progress bar in GUI."""
        print(f"[GUI] Progress: {progress:.2%} - {message}")
        # In real GUI: progress_bar.set_value(progress)

    def update_metrics_ui(metrics: dict[str, float]) -> None:
        """Update metrics display in GUI."""
        print(f"[GUI] Metrics: {metrics}")
        # In real GUI: metrics_panel.update(metrics)

    # Create async consumers
    progress_consumer = create_async_progress_consumer(dispatcher, update_progress_ui)
    metrics_consumer = create_async_metrics_consumer(dispatcher, update_metrics_ui)

    # Run consumers in separate thread with event loop
    def run_consumers() -> None:
        """Run async consumers in background."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        event_loop_holder["loop"] = loop

        # Start consumer tasks - storing in list prevents GC
        _tasks = [
            loop.create_task(progress_consumer()),
            loop.create_task(metrics_consumer()),
        ]

        # Signal that loop is ready
        loop_ready.set()

        loop.run_forever()

    consumer_thread = threading.Thread(target=run_consumers, daemon=True)
    consumer_thread.start()

    # Wait for event loop to initialize (no busy-wait)
    loop_ready.wait(timeout=5.0)

    # Run training with queue-based reporters
    progress_reporter = dispatcher.create_sync_progress_reporter()
    metrics_reporter = dispatcher.create_sync_metrics_reporter()

    try:
        train(
            project,
            progress_reporter=progress_reporter,
            metrics_callback=metrics_reporter,
        )
    finally:
        # Cleanup: close queues and stop event loop
        loop = event_loop_holder["loop"]
        if loop:

            async def cleanup() -> None:
                await dispatcher.close_all()

            # Schedule cleanup and wait for it to complete
            future = asyncio.run_coroutine_threadsafe(cleanup(), loop)
            future.result(timeout=2.0)  # Wait for cleanup to finish

            # Give queues time to flush
            time.sleep(0.1)

            # Stop the event loop
            loop.call_soon_threadsafe(loop.stop)


def example_async_with_gui_helpers(project: Project) -> None:
    """Example using GUI async helpers (recommended for GUI applications).

    This is the recommended pattern for GUI apps, using the AsyncEventLoop
    helper to manage the event loop lifecycle.

    Args:
        project: Project configuration
    """
    from crackz.ai.core.async_callbacks import CallbackDispatcher
    from crackz.ai.training import train
    from crackz.gui.async_helpers import (
        start_gui_callback_consumers,
        stop_gui_callback_consumers,
    )

    # Create dispatcher
    dispatcher = CallbackDispatcher(maxsize=100)

    # Define GUI update callbacks
    def update_progress_ui(progress: float, message: str | None = None) -> None:
        """Update progress bar in GUI."""
        print(f"[GUI] Progress: {progress:.2%} - {message}")

    def update_metrics_ui(metrics: dict[str, float]) -> None:
        """Update metrics display in GUI."""
        print(f"[GUI] Metrics: {metrics}")

    # Start async consumers (manages event loop for you)
    loop, _tasks = start_gui_callback_consumers(dispatcher, update_progress_ui, update_metrics_ui)

    # Run training
    progress_reporter = dispatcher.create_sync_progress_reporter()
    metrics_reporter = dispatcher.create_sync_metrics_reporter()

    try:
        train(
            project,
            progress_reporter=progress_reporter,
            metrics_callback=metrics_reporter,
        )
    finally:
        # Cleanup: stop consumers and close queues
        stop_gui_callback_consumers(dispatcher, loop)


def example_benefits_of_async_pattern() -> None:
    """Demonstrates benefits of the async callback pattern.

    Key benefits:
    1. **Non-blocking**: UI updates don't slow down training
    2. **Backpressure handling**: Queue automatically manages burst updates
    3. **Decoupling**: Training and UI run independently
    4. **Thread-safe**: Queue-based communication is inherently thread-safe
    5. **Backward compatible**: Existing sync callbacks still work

    Performance comparison:
    - Sync callback: Training waits for UI update (~1-5ms per callback)
    - Async callback: Training puts in queue (~0.1ms) and continues
    - For 1000 callbacks/epoch: ~4 seconds saved with async pattern
    """
    print("Async callback benefits:")
    print("1. Non-blocking: UI updates don't slow down training")
    print("2. Backpressure: Queue handles burst updates gracefully")
    print("3. Decoupling: Training and UI run independently")
    print("4. Thread-safe: Queue-based communication")
    print("5. Backward compatible: Sync callbacks still work")


if __name__ == "__main__":
    # Show benefits
    example_benefits_of_async_pattern()

    # Example usage would require a real project
    # from crackz import Project
    # project = Project.load("path/to/project")
    # example_sync_callbacks(project)
    # example_async_with_gui_helpers(project)
