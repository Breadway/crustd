use std::path::Path;

/// Best-effort lid state via the ACPI button interface. Cameras obviously can't see
/// anything useful with the lid shut, so callers should skip capture entirely rather
/// than burn power opening devices against a blocked sensor.
///
/// Returns `false` (assume open) if there's no ACPI lid button on this machine —
/// desktops and some laptops won't have one, and that's not an error condition.
pub fn is_closed() -> bool {
    let Ok(entries) = std::fs::read_dir("/proc/acpi/button/lid") else {
        return false;
    };

    for entry in entries.flatten() {
        let state_path = entry.path().join("state");
        if let Some(state) = read_state(&state_path) {
            if state.contains("closed") {
                return true;
            }
        }
    }

    false
}

fn read_state(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok()
}
