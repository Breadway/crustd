use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tracing::warn;

/// Upper bound on connect + write + read for a single emit. Without this, a bread
/// daemon that accepts the connection but then hangs (deadlocked module, blocked on
/// its own IPC) would stall the presence loop forever, since none of these calls
/// have a timeout of their own.
const IPC_TIMEOUT: Duration = Duration::from_secs(3);

/// Caps how much a single response line can grow before giving up — the peer is
/// local and trusted, but there's no reason to let a misbehaving daemon that never
/// sends a newline grow this buffer without bound.
const MAX_RESPONSE_BYTES: u64 = 4096;

/// Thin client for breadd's line-delimited JSON-RPC IPC socket. Connects fresh for
/// each emit rather than holding a persistent connection — presence changes are rare
/// (debounced, seconds-to-minutes apart), so there's no reason to keep a socket open
/// between them.
pub struct BreadClient {
    socket_path: PathBuf,
    event_prefix: String,
}

impl BreadClient {
    pub fn new(configured_path: &str, event_prefix: String) -> Self {
        let socket_path = if configured_path.is_empty() {
            default_socket_path()
        } else {
            PathBuf::from(configured_path)
        };
        Self {
            socket_path,
            event_prefix,
        }
    }

    /// Emits `<event_prefix>.<suffix>` with the given JSON payload. Errors are logged
    /// and swallowed by the caller's choice — a missing/unreachable bread daemon
    /// shouldn't crash the presence loop. Bounded by `IPC_TIMEOUT` end-to-end so a
    /// daemon that accepts the connection but never responds can't hang the caller.
    pub async fn emit(&self, suffix: &str, data: serde_json::Value) -> Result<()> {
        let event = format!("{}.{}", self.event_prefix, suffix);
        let request = json!({
            "id": "crustd",
            "method": "emit",
            "params": { "event": event, "data": data },
        });
        let line = format!("{}\n", serde_json::to_string(&request)?);

        let response = tokio::time::timeout(IPC_TIMEOUT, self.send_and_wait(&line))
            .await
            .context("bread emit timed out")??;

        if !response_confirms_emitted(&response) {
            warn!(response = %response.trim(), "bread did not confirm emit");
        }

        Ok(())
    }

    async fn send_and_wait(&self, line: &str) -> Result<String> {
        let mut stream = UnixStream::connect(&self.socket_path)
            .await
            .with_context(|| format!("connecting to bread socket at {:?}", self.socket_path))?;

        stream
            .write_all(line.as_bytes())
            .await
            .context("writing emit request")?;

        let mut reader = BufReader::new(stream.take(MAX_RESPONSE_BYTES));
        let mut response = String::new();
        reader
            .read_line(&mut response)
            .await
            .context("reading emit response")?;

        Ok(response)
    }
}

/// Parses the emit response as JSON and checks `emitted == true`, rather than doing
/// a raw substring match — a differently-formatted (but still correct) response,
/// e.g. with extra whitespace or reordered fields, shouldn't be misread as a failure.
///
/// breadd wraps the acknowledgement as `{"id": ..., "result": {"emitted": true}}`,
/// so the flag is read from `result` first and from the top level second. The flat
/// form is kept for older breadd builds; without the nested lookup every successful
/// emit was logged as "bread did not confirm emit" (CRUST-1).
fn response_confirms_emitted(response: &str) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(response) else {
        return false;
    };
    value
        .get("result")
        .and_then(|result| result.get("emitted"))
        .or_else(|| value.get("emitted"))
        .and_then(|emitted| emitted.as_bool())
        .unwrap_or(false)
}

fn default_socket_path() -> PathBuf {
    let runtime_dir = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(runtime_dir).join("bread").join("breadd.sock")
}

#[cfg(test)]
mod tests {
    use super::response_confirms_emitted;

    /// The exact reply breadd sent on hermes, captured in the journal.
    #[test]
    fn nested_result_shape_is_confirmed() {
        assert!(response_confirms_emitted(
            r#"{"id":"crustd","result":{"emitted":true}}"#
        ));
    }

    #[test]
    fn flat_shape_still_works() {
        assert!(response_confirms_emitted(r#"{"emitted":true}"#));
    }

    #[test]
    fn reordered_fields_and_whitespace_are_confirmed() {
        assert!(response_confirms_emitted(
            r#"{
  "result" : {
    "emitted" : true
  },
  "id": "crustd"
}"#
        ));
    }

    #[test]
    fn false_malformed_and_wrong_type_are_not_confirmed() {
        assert!(!response_confirms_emitted(
            r#"{"id":"crustd","result":{"emitted":false}}"#
        ));
        assert!(!response_confirms_emitted(r#"{"emitted":false}"#));
        assert!(!response_confirms_emitted(r#"{"emitted":"true"}"#));
        assert!(!response_confirms_emitted(""));
        assert!(!response_confirms_emitted("not json"));
    }
}
