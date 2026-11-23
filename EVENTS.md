# crustd: bread event integration

crustd is a standalone presence-detection daemon: it works exactly the same
with or without `breadd` running. When breadd *is* present, a debounced
presence transition is published into the shared bread automation fabric.
See the parent `bread` repo's `Documentation.md`, specifically its
"Namespaces" and "Integrating a bread\* app" sections, for the general
convention this follows.

App id: **`crust`**. Transport: crustd's own line-delimited JSON-RPC client
in `src/bread.rs`, which connects to `breadd.sock` for each emit. It does not
link `bread-utils`; the request is a sourced emit (`source: "crust"`) using
the same socket protocol as `BreadClient`.

## Events published (`bread.crust.*`)

| Event | Data | When |
|-------|------|------|
| `bread.crust.changed` | `{ "state": "absent" \| "present" }` | The debounced presence state changed. `present` is emitted after `presence.confirm_frames` consecutive positive IR bursts; `absent` after `presence.release_frames` consecutive negative bursts. Not emitted at startup, on every poll tick, for motion without a face confirmation, or when the state remains unchanged. |

The payload carries only the state. It never includes camera paths, device
identifiers, face-detection confidence, frame data, or any biometric
information.

## Commands honored (`bread.command.crust.*`)

None. crustd does not subscribe to the command bus; it is a one-way event
publisher.

## Not implemented

- No `bread.command.crust.*` verb exists. Presence detection has no
  remote-triggered action, so there is nothing to stub.
- No events are published for per-frame confidence, motion, lid state, idle
  state, or historical presence. Those are internal signals and logs, not
  bus state.
- No payload field identifies the camera, the detected person, or the model.
  Adding one would expose local hardware and biometric detail without a
  consumer that needs it.

## Fail-safe behavior

- If breadd is not installed or not running, the emit attempt fails, crustd
  logs a warning, and the presence loop continues unchanged. Presence
  detection and any local policy built on it never depend on the bus.
- Every emit has a three-second upper bound covering connect, write, and
  response, so a breadd that accepts the connection and then hangs cannot
  stall the presence loop.
- Each transition opens a fresh short-lived connection; there is no
  persistent socket to reconnect. A breadd restart is picked up by the next
  transition automatically.
- If the running breadd does not know the `crust` app id, it rejects the
  sourced emit. crustd logs the unconfirmed response and continues; an older
  daemon never changes presence behavior.
