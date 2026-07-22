use std::path::PathBuf;

use anyhow::{Context, Result};
use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tracing::warn;

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
    /// shouldn't crash the presence loop.
    pub async fn emit(&self, suffix: &str, data: serde_json::Value) -> Result<()> {
        let event = format!("{}.{}", self.event_prefix, suffix);
        let mut stream = UnixStream::connect(&self.socket_path)
            .await
            .with_context(|| format!("connecting to bread socket at {:?}", self.socket_path))?;

        let request = json!({
            "id": "crustd",
            "method": "emit",
            "params": { "event": event, "data": data },
        });
        let line = format!("{}\n", serde_json::to_string(&request)?);
        stream
            .write_all(line.as_bytes())
            .await
            .context("writing emit request")?;

        let mut reader = BufReader::new(stream);
        let mut response = String::new();
        reader
            .read_line(&mut response)
            .await
            .context("reading emit response")?;

        if !response.contains("\"emitted\":true") {
            warn!(response = %response.trim(), "bread did not confirm emit");
        }

        Ok(())
    }
}

fn default_socket_path() -> PathBuf {
    let runtime_dir = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(runtime_dir).join("bread").join("breadd.sock")
}
