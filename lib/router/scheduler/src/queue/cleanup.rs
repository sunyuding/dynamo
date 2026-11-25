// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Releasing bookings: the handles and leases callers hold, the cleanup queue their drops
//! feed, and the actor's drain of it.

use super::*;

#[derive(Debug)]
pub(super) enum SchedulerCleanupTarget {
    AdmissionLifecycle(Arc<AdmissionLifecycleTransfer>),
    Booking(SchedulerBookingDescriptor),
    ExpiredBooking(SchedulerBookingDescriptor),
}

#[derive(Debug)]
enum AdmissionLifecycleState {
    Unarmed { request_id: String },
    Pending { request_id: String },
    Booking(SchedulerBookingDescriptor),
    Disarmed,
}

#[derive(Debug)]
pub(super) struct AdmissionLifecycleTransfer {
    state: Mutex<AdmissionLifecycleState>,
}

pub(super) enum AdmissionLifecycleCleanup {
    Pending { request_id: String },
    Booking(SchedulerBookingDescriptor),
}

impl AdmissionLifecycleTransfer {
    pub(super) fn new(request_id: String) -> Self {
        Self {
            state: Mutex::new(AdmissionLifecycleState::Unarmed { request_id }),
        }
    }

    pub(super) fn arm_pending(&self) {
        let mut state = self.state.lock();
        let AdmissionLifecycleState::Unarmed { request_id } = &*state else {
            return;
        };
        *state = AdmissionLifecycleState::Pending {
            request_id: request_id.clone(),
        };
    }

    pub(super) fn arm_booking(&self, booking: SchedulerBookingDescriptor) {
        let mut state = self.state.lock();
        if matches!(
            *state,
            AdmissionLifecycleState::Unarmed { .. } | AdmissionLifecycleState::Pending { .. }
        ) {
            *state = AdmissionLifecycleState::Booking(booking);
        }
    }

    fn disarm(&self) {
        *self.state.lock() = AdmissionLifecycleState::Disarmed;
    }

    pub(super) fn cleanup_target(&self) -> Option<AdmissionLifecycleCleanup> {
        let mut state = self.state.lock();
        let cleanup = match &*state {
            AdmissionLifecycleState::Unarmed { .. } | AdmissionLifecycleState::Disarmed => None,
            AdmissionLifecycleState::Pending { request_id } => {
                Some(AdmissionLifecycleCleanup::Pending {
                    request_id: request_id.clone(),
                })
            }
            AdmissionLifecycleState::Booking(booking) => {
                Some(AdmissionLifecycleCleanup::Booking(booking.clone()))
            }
        };
        *state = AdmissionLifecycleState::Disarmed;
        cleanup
    }

    pub(super) fn is_armed(&self) -> bool {
        matches!(
            *self.state.lock(),
            AdmissionLifecycleState::Pending { .. } | AdmissionLifecycleState::Booking(_)
        )
    }
}

pub(super) struct AdmissionCleanupEntry {
    pub(super) target: SchedulerCleanupTarget,
}

#[derive(Default)]
pub(super) struct AdmissionCleanup {
    dirty: SegQueue<AdmissionCleanupEntry>,
    pending: AtomicBool,
}

impl AdmissionCleanup {
    pub(super) fn enqueue(&self, cleanup: AdmissionCleanupEntry) -> bool {
        self.dirty.push(cleanup);
        !self.pending.swap(true, AtomicOrdering::AcqRel)
    }

    fn drain(&self) -> Vec<AdmissionCleanupEntry> {
        if !self.pending.load(AtomicOrdering::Acquire) {
            return Vec::new();
        }

        // Drain to a quiescent point to preserve the coalesced-wake handoff. A large burst of
        // active lease drops can delay actor commands; any bounded or interleaved drain
        // must preserve wake correctness and be benchmarked.
        let mut dirty = Vec::new();
        loop {
            while let Some(cleanup) = self.dirty.pop() {
                dirty.push(cleanup);
            }
            self.pending.store(false, AtomicOrdering::Release);
            if self.dirty.is_empty() {
                return dirty;
            }
            self.pending.store(true, AtomicOrdering::Release);
        }
    }
}

fn enqueue_cleanup_entry(
    cleanup: &AdmissionCleanup,
    actor_tx: &mpsc::Sender<AdmissionCommand>,
    entry: AdmissionCleanupEntry,
) {
    if !cleanup.enqueue(entry) {
        return;
    }
    match actor_tx.try_send(AdmissionCommand::Cleanup) {
        Ok(()) | Err(mpsc::error::TrySendError::Full(_)) => {}
        Err(mpsc::error::TrySendError::Closed(_)) => {
            // No actor remains to consume destructor-only cleanup.
            cleanup.drain();
        }
    }
}

/// Cloneable non-blocking ingress for attempt-scoped scheduler cleanup.
#[derive(Clone)]
pub struct SchedulerBookingCleanup {
    cleanup: Arc<AdmissionCleanup>,
    actor_tx: mpsc::Sender<AdmissionCommand>,
    direct_release: DirectBookingRelease,
}

pub(super) type DirectBookingRelease =
    Arc<dyn Fn(&SchedulerBookingDescriptor) -> Result<(), SequenceError> + Send + Sync>;

/// The scheduler's acknowledgement of an enqueued booking release; `wait` resolves with
/// the release outcome.
pub struct SchedulerCleanupAck {
    response: Result<(), SequenceError>,
}

#[cfg(any(test, feature = "testing"))]
impl SchedulerCleanupAck {
    /// The cleanup outcome, for tests that assert on acknowledgement.
    pub fn response(&self) -> &Result<(), SequenceError> {
        &self.response
    }
}

impl SchedulerCleanupAck {
    pub async fn wait(self) -> Result<(), SequenceError> {
        self.response
    }
}

impl SchedulerBookingCleanup {
    fn enqueue_entry(&self, entry: AdmissionCleanupEntry) {
        enqueue_cleanup_entry(&self.cleanup, &self.actor_tx, entry);
    }

    pub fn enqueue(&self, booking: SchedulerBookingDescriptor) {
        self.enqueue_entry(AdmissionCleanupEntry {
            target: SchedulerCleanupTarget::Booking(booking),
        });
    }

    pub fn enqueue_expired(&self, booking: SchedulerBookingDescriptor) {
        self.enqueue_entry(AdmissionCleanupEntry {
            target: SchedulerCleanupTarget::ExpiredBooking(booking),
        });
    }

    /// Apply explicit release now. The acknowledgement covers the mutation and
    /// capacity notification; pending admission runs asynchronously.
    pub fn enqueue_acknowledged(&self, booking: SchedulerBookingDescriptor) -> SchedulerCleanupAck {
        SchedulerCleanupAck {
            response: (self.direct_release)(&booking),
        }
    }

    pub async fn enqueue_and_wait(
        &self,
        booking: SchedulerBookingDescriptor,
    ) -> Result<(), SequenceError> {
        self.enqueue_acknowledged(booking).wait().await
    }
}

/// A booking whose release this handle owns until `commit` hands it over.
/// Dropping an armed handle frees the booking through the scheduler's cleanup queue.
#[must_use]
pub struct BookingHandle {
    pub(super) booking: SchedulerBookingDescriptor,
    cleanup: SchedulerBookingCleanup,
    armed: bool,
}

impl std::fmt::Debug for BookingHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BookingHandle")
            .field("booking", &self.booking)
            .field("armed", &self.armed)
            .finish()
    }
}

impl BookingHandle {
    /// Hand the booking to a longer-lived owner; the handle stops guarding it.
    #[must_use]
    pub fn commit(mut self) -> SchedulerBookingDescriptor {
        self.armed = false;
        SchedulerBookingDescriptor {
            request_id: std::mem::take(&mut self.booking.request_id),
            worker: self.booking.worker,
            attempt_id: self.booking.attempt_id,
        }
    }

    /// Free the booking directly and record a capacity notification.
    /// This does not wait for the actor to recheck pending admissions.
    pub async fn release(mut self) -> Result<(), SequenceError> {
        self.armed = false;
        self.cleanup
            .enqueue_acknowledged(self.booking.clone())
            .wait()
            .await
    }
}

impl Drop for BookingHandle {
    fn drop(&mut self) {
        if self.armed {
            self.cleanup.enqueue(self.booking.clone());
        }
    }
}

/// Single-owner cleanup lease for one scheduler-tracked request.
pub struct RequestLifecycleLease {
    cleanup: Arc<AdmissionCleanup>,
    actor_tx: mpsc::Sender<AdmissionCommand>,
    pub(super) transfer: Option<Arc<AdmissionLifecycleTransfer>>,
}

impl std::fmt::Debug for RequestLifecycleLease {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RequestLifecycleLease")
            .field("transfer", &self.transfer)
            .finish_non_exhaustive()
    }
}

impl RequestLifecycleLease {
    pub(crate) fn disarm(&mut self) {
        if let Some(transfer) = self.transfer.take() {
            transfer.disarm();
        }
    }

    /// Hand the booking to its long-term owner: the lease stops guarding it.
    #[must_use]
    pub fn commit(mut self) -> Option<SchedulerBookingDescriptor> {
        let booking = match self.transfer.as_ref().map(|transfer| transfer.state.lock()) {
            Some(state) => match &*state {
                AdmissionLifecycleState::Booking(booking) => Some(booking.clone()),
                _ => None,
            },
            None => None,
        };
        self.disarm();
        booking
    }
}

impl Drop for RequestLifecycleLease {
    fn drop(&mut self) {
        let Some(transfer) = self.transfer.take() else {
            return;
        };
        if !transfer.is_armed() {
            return;
        }
        enqueue_cleanup_entry(
            &self.cleanup,
            &self.actor_tx,
            AdmissionCleanupEntry {
                target: SchedulerCleanupTarget::AdmissionLifecycle(transfer),
            },
        );
    }
}

impl<
    P: SequencePublisher + 'static,
    C: WorkerConfigLike + Send + Sync + 'static,
    Sel: WorkerSelector<C> + Send + 'static,
    RF: OverlapScoresRefresh + Send + Sync + 'static,
> SchedulerQueue<P, C, Sel, RF>
{
    pub(crate) fn new_request_lifecycle_lease(
        &self,
        request_id: Option<&str>,
    ) -> Option<Box<RequestLifecycleLease>> {
        let request_id = request_id?;
        Some(Box::new(RequestLifecycleLease {
            cleanup: Arc::clone(&self.cleanup),
            actor_tx: self.admission_tx.clone(),
            transfer: Some(Arc::new(AdmissionLifecycleTransfer::new(
                request_id.to_string(),
            ))),
        }))
    }

    pub fn booking_cleanup(&self) -> SchedulerBookingCleanup {
        SchedulerBookingCleanup {
            cleanup: Arc::clone(&self.cleanup),
            actor_tx: self.admission_tx.clone(),
            direct_release: Arc::clone(&self.direct_release),
        }
    }

    /// An armed handle for an existing booking.
    pub(crate) fn booking_handle(&self, booking: SchedulerBookingDescriptor) -> BookingHandle {
        BookingHandle {
            booking,
            cleanup: self.booking_cleanup(),
            armed: true,
        }
    }
}

impl<
    P: SequencePublisher + 'static,
    C: WorkerConfigLike + Send + Sync + 'static,
    Sel: WorkerSelector<C> + Send + 'static,
    RF: OverlapScoresRefresh + Send + Sync + 'static,
> SchedulerQueueActor<P, C, Sel, RF>
{
    pub(super) fn drain_cleanup(&mut self) -> bool {
        let dirty = self.cleanup.drain();
        if dirty.is_empty() {
            return false;
        }

        let mut made_ready = false;
        let mut removed_ready_head = false;
        let mut unmanaged_request_ids = HashSet::new();
        for cleanup in dirty {
            match cleanup.target {
                SchedulerCleanupTarget::AdmissionLifecycle(transfer) => {
                    let result = match transfer.cleanup_target() {
                        Some(AdmissionLifecycleCleanup::Pending { request_id }) => {
                            unmanaged_request_ids.insert(request_id);
                            Ok(())
                        }
                        Some(AdmissionLifecycleCleanup::Booking(booking)) => self
                            .slots
                            .free_if_booking(
                                &booking.request_id,
                                booking.worker,
                                booking.attempt_id,
                                Instant::now(),
                            )
                            .map(|outcome| {
                                if outcome.is_applied() {
                                    made_ready = true;
                                }
                            }),
                        None => Ok(()),
                    };
                    if let Err(error) = result {
                        tracing::error!(%error, "Failed to release scheduler admission lifecycle");
                    }
                }
                SchedulerCleanupTarget::Booking(booking) => {
                    let result = self
                        .slots
                        .free_if_booking(
                            &booking.request_id,
                            booking.worker,
                            booking.attempt_id,
                            Instant::now(),
                        )
                        .map(|outcome| {
                            if outcome.is_applied() {
                                made_ready = true;
                            }
                        });
                    if let Err(error) = result {
                        tracing::error!(
                            request_id = %booking.request_id,
                            worker = ?booking.worker,
                            attempt_id = %booking.attempt_id,
                            %error,
                            "Failed to release request-attempt scheduler booking"
                        );
                    }
                }
                SchedulerCleanupTarget::ExpiredBooking(booking) => {
                    let result = self
                        .slots
                        .expire_if_booking(
                            &booking.request_id,
                            booking.worker,
                            booking.attempt_id,
                            Instant::now(),
                        )
                        .map(|outcome| {
                            if outcome.is_applied() {
                                made_ready = true;
                            }
                        });
                    if let Err(error) = result {
                        tracing::error!(
                            request_id = %booking.request_id,
                            worker = ?booking.worker,
                            attempt_id = %booking.attempt_id,
                            %error,
                            "Failed to expire request-attempt scheduler booking"
                        );
                    }
                }
            }
        }
        if !unmanaged_request_ids.is_empty() {
            for class_index in 0..self.profile.classes().len() {
                let (removed, class_head_removed) =
                    self.pending.take_if_in_class(class_index, |queued| {
                        queued
                            .request
                            .mode
                            .tracked_request_id()
                            .is_some_and(|request_id| unmanaged_request_ids.contains(request_id))
                    });
                removed_ready_head |= class_head_removed;
                let lost_entries = !removed.is_empty();
                for entry in removed {
                    self.subtract_pending_counters(class_index, entry.snapshot());
                }
                if lost_entries {
                    self.rearm_class_timer(class_index);
                }
            }
        }
        made_ready || (removed_ready_head && self.has_dispatchable_ready_head())
    }
}
