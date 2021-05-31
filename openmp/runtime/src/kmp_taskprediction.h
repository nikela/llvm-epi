#ifndef KMP_TASKPREDICTION_H
#define KMP_TASKPREDICTION_H

#if LIBOMP_TASK_PREDICTION

#include "kmp.h"

void __kmp_update_task_elapsed_time(kmp_int32 label, kmp_int32 cost,
                                    kmp_uint64 elapsed);

// Monotonic clock.
kmp_uint64 __kmp_task_time();

enum kmp_task_pred_state {
  KTP_NONE = 0,
  KTP_READY,
  KTP_EXECUTE,
};

// A label of -1 means a task without label.
// FIXME: Devise a mechanism to associate labels to locations if there is no
// label.
constexpr kmp_int32 kmp_default_task_label = -1;
constexpr kmp_int32 kmp_default_task_cost = 1;

void __kmp_task_change_state(kmp_team_t *team, kmp_task_pred_state old_state,
                             kmp_task_pred_state new_state, kmp_int32 label,
                             kmp_int32 cost, kmp_int32 *prediction_info);

kmp_int32 __kmp_predicted_threads();

#if LIBOMP_TASK_PREDICTION_TIMERS
// Only used by MAIN thread
void __kmp_prediction_start_timer();
void __kmp_prediction_stop_timer();
#endif

#endif

#endif // KMP_TASKPREDICTION_H
