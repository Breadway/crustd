use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

use wayland_client::globals::{registry_queue_init, GlobalListContents};
use wayland_client::protocol::{wl_registry, wl_seat};
use wayland_client::protocol::wl_seat::WlSeat;
use wayland_client::{Connection, Dispatch, QueueHandle};
use wayland_protocols::ext::idle_notify::v1::client::ext_idle_notification_v1::{
    self, ExtIdleNotificationV1,
};
use wayland_protocols::ext::idle_notify::v1::client::ext_idle_notifier_v1::ExtIdleNotifierV1;

/// Tracks whether the compositor considers the seat's input (keyboard/mouse/etc.)
/// idle, via the standard `ext-idle-notify-v1` protocol (the same one `hypridle`
/// itself relies on). Runs its own connection on a dedicated thread since the
/// wayland-client event loop is blocking.
pub struct IdleWatcher {
    idle: Arc<AtomicBool>,
}

impl IdleWatcher {
    /// Returns `None` if the compositor doesn't support ext-idle-notify-v1 (or the
    /// connection otherwise fails) — callers should treat that as "can't use this
    /// signal" and fall back to the camera-only behavior rather than erroring out.
    pub fn spawn(timeout_ms: u32) -> Option<Self> {
        let idle = Arc::new(AtomicBool::new(false));
        let idle_for_thread = Arc::clone(&idle);
        let (ready_tx, ready_rx) = mpsc::channel();

        std::thread::Builder::new()
            .name("crustd-idle-notify".into())
            .spawn(move || run(timeout_ms, idle_for_thread, ready_tx))
            .ok()?;

        ready_rx.recv().ok()?.then_some(Self { idle })
    }

    pub fn is_idle(&self) -> bool {
        self.idle.load(Ordering::Relaxed)
    }
}

struct State {
    idle: Arc<AtomicBool>,
}

fn run(timeout_ms: u32, idle: Arc<AtomicBool>, ready_tx: mpsc::Sender<bool>) {
    let Ok(conn) = Connection::connect_to_env() else {
        let _ = ready_tx.send(false);
        return;
    };

    let Ok((globals, mut queue)) = registry_queue_init::<State>(&conn) else {
        let _ = ready_tx.send(false);
        return;
    };
    let qh = queue.handle();

    let seat = globals.bind::<WlSeat, _, _>(&qh, 1..=8, ());
    let notifier = globals.bind::<ExtIdleNotifierV1, _, _>(&qh, 1..=1, ());

    let (Ok(seat), Ok(notifier)) = (seat, notifier) else {
        let _ = ready_tx.send(false);
        return;
    };

    let mut state = State { idle };
    let _notification = notifier.get_idle_notification(timeout_ms, &seat, &qh, ());

    let _ = ready_tx.send(true);

    loop {
        if queue.blocking_dispatch(&mut state).is_err() {
            // Compositor connection dropped; stick with the last known idle state
            // rather than spinning a busy loop.
            break;
        }
    }
}

impl Dispatch<ExtIdleNotificationV1, ()> for State {
    fn event(
        state: &mut Self,
        _proxy: &ExtIdleNotificationV1,
        event: ext_idle_notification_v1::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        match event {
            ext_idle_notification_v1::Event::Idled => state.idle.store(true, Ordering::Relaxed),
            ext_idle_notification_v1::Event::Resumed => state.idle.store(false, Ordering::Relaxed),
            _ => {}
        }
    }
}

// No-op dispatch impls for objects whose events we don't care about.
impl Dispatch<WlSeat, ()> for State {
    fn event(_: &mut Self, _: &WlSeat, _: wl_seat::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {}
}

impl Dispatch<ExtIdleNotifierV1, ()> for State {
    fn event(
        _: &mut Self,
        _: &ExtIdleNotifierV1,
        _: <ExtIdleNotifierV1 as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wl_registry::WlRegistry, GlobalListContents> for State {
    fn event(
        _: &mut Self,
        _: &wl_registry::WlRegistry,
        _: wl_registry::Event,
        _: &GlobalListContents,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}
