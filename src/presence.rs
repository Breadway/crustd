use crate::config::PresenceConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresenceState {
    Absent,
    Present,
}

impl PresenceState {
    pub fn as_str(&self) -> &'static str {
        match self {
            PresenceState::Absent => "absent",
            PresenceState::Present => "present",
        }
    }
}

/// Debounced presence state machine. Feed it one boolean per IR burst ("was a face
/// seen above threshold") and it reports state transitions only after `confirm_frames`
/// consecutive positives (to flip on) or `release_frames` consecutive negatives (to
/// flip off) — this is what keeps a single flickery frame from toggling the state.
pub struct PresenceTracker {
    state: PresenceState,
    positive_streak: u32,
    negative_streak: u32,
    confirm_frames: u32,
    release_frames: u32,
}

impl PresenceTracker {
    pub fn new(cfg: &PresenceConfig) -> Self {
        Self {
            state: PresenceState::Absent,
            positive_streak: 0,
            negative_streak: 0,
            confirm_frames: cfg.confirm_frames.max(1),
            release_frames: cfg.release_frames.max(1),
        }
    }

    pub fn state(&self) -> PresenceState {
        self.state
    }

    /// Count of consecutive failed retention checks since presence was last
    /// confirmed. Zero means the last check (if any) was positive, or none has run
    /// yet — callers use this to decide whether to use the slow stillness-timeout
    /// cadence or the fast retry-after-miss cadence for the next IR check.
    pub fn negative_streak(&self) -> u32 {
        self.negative_streak
    }

    /// Returns `Some(new_state)` only when the debounced state actually changes.
    pub fn observe(&mut self, face_seen: bool) -> Option<PresenceState> {
        if face_seen {
            self.positive_streak += 1;
            self.negative_streak = 0;
        } else {
            self.negative_streak += 1;
            self.positive_streak = 0;
        }

        match self.state {
            PresenceState::Absent if self.positive_streak >= self.confirm_frames => {
                self.state = PresenceState::Present;
                Some(self.state)
            }
            PresenceState::Present if self.negative_streak >= self.release_frames => {
                self.state = PresenceState::Absent;
                Some(self.state)
            }
            _ => None,
        }
    }
}
