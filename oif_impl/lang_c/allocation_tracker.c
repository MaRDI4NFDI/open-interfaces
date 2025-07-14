#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "oif/util.h"

#include "allocation_tracker.h"

#define ALLOCATION_TRACKER_TYPE 120

static int INITIAL_CAPACITY_ = 10;

struct allocation_tracker_t {
    int type;
    void **pointers;
    cleanup_fn **cleanup_fns;
    size_t size;
    size_t capacity;
};

AllocationTracker *
allocation_tracker_init(void)
{
    AllocationTracker *tracker = oif_util_malloc(sizeof(AllocationTracker));
    if (tracker == NULL) {
        goto report_error_and_exit;
    }

    tracker->type = ALLOCATION_TRACKER_TYPE;
    tracker->pointers = oif_util_malloc(sizeof(*tracker->pointers) * INITIAL_CAPACITY_);
    if (tracker->pointers == NULL) {
        goto report_error_and_exit;
    }
    tracker->cleanup_fns = oif_util_malloc(sizeof(*tracker->cleanup_fns) * INITIAL_CAPACITY_);
    tracker->size = 0;
    tracker->capacity = INITIAL_CAPACITY_;

    return tracker;

report_error_and_exit:
    fprintf(stderr, "Could not allocate memory for allocation tracker. So ironic\n");
    exit(1);
}

void
allocation_tracker_add(AllocationTracker *tracker, void *pointer, cleanup_fn *func)
{
    if (pointer == NULL) {
        fprintf(stderr, "Cannot track a NULL pointer\n");
        exit(1);
    }

    if (func == NULL) {
        // If function is not provided, we use `free` from `<stdlib.h>`
        // as the tracked object is something simple.
        func = free;
    }

    tracker->pointers[tracker->size] = pointer;
    tracker->cleanup_fns[tracker->size] = func;
    tracker->size++;

    if (tracker->size == tracker->capacity) {
        fprintf(stderr, "WARNING: Allocation tracker capacity has reached its maximum\n");
    }
}

void
allocation_tracker_free(void *tracker_)
{
    AllocationTracker *tracker = tracker_;
    assert(tracker->type == ALLOCATION_TRACKER_TYPE);
    for (size_t i = 0; i < tracker->size; ++i) {
        tracker->cleanup_fns[i](tracker->pointers[i]);
    }

    oif_util_free(tracker->pointers);
    oif_util_free(tracker->cleanup_fns);
    oif_util_free(tracker);
    tracker = NULL;
}
