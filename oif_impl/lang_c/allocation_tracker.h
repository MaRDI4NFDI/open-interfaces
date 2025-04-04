#pragma once

typedef void(cleanup_fn)(void *p);

typedef struct allocation_tracker_t AllocationTracker;

/**
 * Initialize AllocationTracker object.
 */
AllocationTracker *
allocation_tracker_init(void);

/**
 * Add a pointer along with the cleanup function to the tracker.
 * If the cleanup function must be `stdlib` free, then pass `NULL`.
 * Returns the value of the added pointer.
 */
void
allocation_tracker_add(AllocationTracker *tracker, void *pointer, cleanup_fn *func);

/**
 * Get the last added pointer.
 */
void *
allocation_tracker_get_current_pointer(AllocationTracker *tracker);

/**
 * Invoke cleanup functions on tracked pointers and release the memory
 * for the tracker itself.
 */
void
allocation_tracker_free(void *tracker);
