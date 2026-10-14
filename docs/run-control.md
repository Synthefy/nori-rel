# Progress and cancellation

`predict()` runs synchronously. Pass `on_event` to observe stages and the final
status; events are delivered on the calling thread:

```python
frame = client.predict(**request, on_event=lambda event: print(event.message))
summary = frame.attrs["run"]
```

Here `request` contains the same source, query, and time arguments as a normal
[PQL prediction](pql.md). To run on a worker thread:

```python
from nori_rel import PredictionCancelled, RunTimeout

run = client.start_predict(**request)
try:
    frame = run.result(timeout=60)
except RunTimeout:
    run.cancel()
    try:
        run.result()  # wait for cancellation and cleanup
    except PredictionCancelled:
        pass
```

`start_predict()` accepts prediction arguments except `return_details`.
`run.result()` returns the prediction frame, including metadata in `attrs`.
A timeout only ends the caller's wait. Cancellation is cooperative at execution
boundaries; it cannot guarantee immediate interruption of an in-flight database
query or model operation. SQL statement timeouts provide a separate bound on
queries. Handle `PredictionFailed` for a background prediction that failed.

Iterate `run.events()` to replay events from the start and follow new ones.
`run.events(after=event_id)` resumes after an event. `on_event` on a background
run executes on the worker thread, so UI callbacks must arrange their own thread
handoff. Do not put credentials or data values into application progress logs.

`run.pause()` and `run.resume()` take effect at supported boundaries. Commands
return receipts; a request to a finished run is rejected. Pausing cannot suspend
an already running database or model call. `client.get_run(run.run_id)` retrieves
a run owned by that client.

Use a context manager for the client. Normal `close()` waits for active runs
and closes database pools; a paused run is cancelled so close can make progress.
`close(timeout_seconds=...)` requests cancellation after its waiting period, then
waits for cooperative cleanup. This is not a hard process deadline. Exiting the
client context with an exception also requests cancellation.
