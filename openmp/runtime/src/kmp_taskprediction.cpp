/*
 * kmp_prediction.cpp -- Tasking prediction routines.
 */

#include "kmp.h"
#include "kmp_lock.h"

#if LIBOMP_TASK_PREDICTION

#include "kmp_wait_release.h"
#include "kmp_taskprediction.h"
#include <algorithm>
#include <ctime>
#include <map>
#include <atomic>
#include <memory>
#include <cmath>
#include <time.h>

// Uncomment this for extrae support
#define TASKPREDICTION_EXTRAE 1

// 100 microseconds. FIXME: make this user-settable.
enum { TS_STALE_NS = 100000 };

// This is not elegant but not a thing this runtime is not already doing
// elsewhere, so we are not worse.
template <typename T> std::atomic<T> &to_atomic(T &t) {
  return reinterpret_cast<std::atomic<T> &>(t);
}

kmp_uint64 timespec_to_nanoseconds(struct timespec ts) {
  return ts.tv_sec * 1000000000 + ts.tv_nsec;
}

struct timespec __kmp_get_coarse_timespec() {
  struct timespec n;
  int res = clock_gettime(CLOCK_MONOTONIC_COARSE, &n);
  if (UNLIKELY(res == -1)) {
    n.tv_sec = 0;
    n.tv_nsec = 0;
  }
  return n;
}

struct timespec __kmp_get_precise_timespec() {
  struct timespec n;
  int res = clock_gettime(CLOCK_MONOTONIC, &n);
  if (UNLIKELY(res == -1)) {
    n.tv_sec = 0;
    n.tv_nsec = 0;
  }
  return n;
}

kmp_uint64 __kmp_task_time() {
  return timespec_to_nanoseconds(__kmp_get_precise_timespec());
}

// Add-only queue.
struct WindowData {
  // FIXME: Make this user settable.
  // 20 is the default value in the reference implementation.
  static const int window_size = 20;

  unsigned num = 0;
  unsigned next = 0;
  double window[window_size];
  double rolling_mean = 0.0;

  double add(double v) {
    if (num < window_size) {
      num++;
      rolling_mean = (rolling_mean * (num - 1) + v) / num;
    } else {
      rolling_mean += (-window[next] + v) / window_size;
    }
    KA_TRACE(100, ("Window: Adding %e to prediction. Rolling mean is %e\n", v,
                 rolling_mean));
    window[next] = v;
    next++;
    if (next == window_size) {
      next = 0;
    }
    return rolling_mean;
  }

  double get_rolling_mean() const { return rolling_mean; }
  bool empty() const { return num == 0; }
};

struct TASLock {
private:
  kmp_tas_lock_t m;

public:
  TASLock() { __kmp_init_tas_lock(&m); }
  ~TASLock() { __kmp_destroy_tas_lock(&m); }
  TASLock(const TASLock &) = delete;
  TASLock(TASLock &&) = delete;
  TASLock &operator=(const TASLock &) = delete;
  TASLock &operator=(TASLock &&) = delete;

  void lock(kmp_int32 gid) noexcept { __kmp_acquire_tas_lock(&m, gid); }
  void unlock(kmp_int32 gid) noexcept { __kmp_release_tas_lock(&m, gid); }
  bool try_lock(kmp_int32 gid) noexcept { return __kmp_test_tas_lock(&m, gid); }
};

struct TASLockGuard {
private:
  TASLock &m;
  kmp_int32 g;

public:
  TASLockGuard(const TASLock &m)
      : m(const_cast<TASLock &>(m)), g(__kmp_get_gtid()) {
    this->m.lock(g);
  }
  ~TASLockGuard() { m.unlock(g); }
};

struct Window : WindowData {
private:
  TASLock m;
public:
  // We don't copy the lock which is only to protect our owned WindowData.
  Window() = default;
  Window(const Window& w) : WindowData(w) { }
  Window(const Window&& w) noexcept : WindowData(std::move(w)) { }
  Window &operator=(const Window &w) {
    if (this != &w) {
      WindowData::operator=(w);
    }
    return *this;
  }
  Window &operator=(const Window &&w) {
    if (this != &w) {
      WindowData::operator=(std::move(w));
    }
    return *this;
  }
  ~Window() { }

  double add(double v) {
    TASLockGuard T(m);
    return WindowData::add(v);
  }

  double get_rolling_mean_unlocked() const {
    return WindowData::get_rolling_mean();
  }

  double get_rolling_mean() const {
    TASLockGuard T(m);
    return get_rolling_mean_unlocked();
  }

  bool empty() const {
    TASLockGuard T(m);
    return WindowData::empty();
  }
};

struct PredictionTaskType {
  kmp_uint64 ready_wl = 0;
  kmp_uint64 exec_wl = 0;
  kmp_uint64 num_ready = 0;
  kmp_uint64 num_executing = 0;
  kmp_uint64 num_without_prediction = 0;

  Window cost_per_time;
};

// We use map to avoid pointer invalidation even if std::map is not very
// efficient.
using TaskTypeToPrediction = std::map<kmp_int32, PredictionTaskType>;

struct PredictionData {
  TASLock m; // Protects concurrent updates to task_type_to_pred.
  TaskTypeToPrediction task_type_to_pred;

  // Rolling window of earlier thread predictions, has its own mutex.
  Window thread_prediction;


  PredictionData() { }
  ~PredictionData() { }

#if LIBOMP_TASK_PREDICTION_TIMERS
  static void prediction_signal_handler(int sig, siginfo_t *si, void *uc);

  // Only used by the main thread.
  timer_t prediction_timer;
  bool timer_init = false;

  void create_timer_if_needed();
  void start_timer();
  void stop_timer();
#endif
};

#if LIBOMP_TASK_PREDICTION_TIMERS
void PredictionData::create_timer_if_needed() {
  if (timer_init)
    return;
  timer_init = true;
  // FIXME - This is OS specific and should be encapsulated elsewhere
  // Based on the example from timer_create(2) man page
  /* Establish handler for timer signal. */

// FIXME - Do this in a better way.
#define CLOCKID CLOCK_REALTIME
#define SIG (SIGRTMIN + 8)
  KMP_DEBUG_ASSERT(SIG < SIGRTMAX && "Overflow in rt signal");

  struct sigevent sev;
  struct sigaction sa;
  int status;

  sa.sa_flags = SA_SIGINFO;
  sa.sa_sigaction = prediction_signal_handler;
  sigemptyset(&sa.sa_mask);
  status = sigaction(SIG, &sa, NULL);
  KMP_CHECK_SYSFAIL("sigaction", status);

  // /* Block timer signal temporarily. */
  // sigemptyset(&mask);
  // sigaddset(&mask, SIG);
  // status = sigprocmask(SIG_SETMASK, &mask, NULL);
  // KMP_CHECK_SYSFAIL("sigprocmask", status);

  /* Create the timer. */
  sev.sigev_notify = SIGEV_THREAD_ID;
  sev.sigev_signo = SIG;
  // Note: intentionally getting the TID of the process
  // FIXME: go through the team and gather this information.
  // FIXME: this is wrong if the main thread of the process is not the main
  // thread of OpenMP.
  // Field _tid is described in the man as sigev_notify_thread_id but given
  // that this is Linux specific its name is likely nonstable.
  sev._sigev_un._tid = getpid();
  status = timer_create(CLOCKID, &sev, &prediction_timer);
  KMP_CHECK_SYSFAIL("timer_create", status);
}

void PredictionData::start_timer() {
  create_timer_if_needed();

  int status;
  struct itimerspec its;
  its.it_value.tv_sec = 0;
  its.it_value.tv_nsec = TS_STALE_NS;
  its.it_interval.tv_sec = its.it_value.tv_sec;
  its.it_interval.tv_nsec = its.it_value.tv_nsec;

  status = timer_settime(&prediction_timer, 0, &its, NULL);
  KMP_CHECK_SYSFAIL("timer_settime", status);
}

void PredictionData::stop_timer()
{
  int status;
  // Disarm timer.
  struct itimerspec its = { };

  status =  timer_settime(&prediction_timer, 0, &its, NULL);
  KMP_CHECK_SYSFAIL("timer_settime", status);
}

void PredictionData::prediction_signal_handler(int sig, siginfo_t *si, void *uc)
{
  // Do nothing
  __kmp_get_or_update_threads_prediction();
}
#endif

kmp_int32 __kmp_get_or_update_threads_prediction();

// Initialized to zero.
std::atomic<PredictionData *> __kmp_task_predictions_g;

PredictionData *__kmp_get_or_allocate_prediction_data() {
  PredictionData *result = __kmp_task_predictions_g.load();
  if (UNLIKELY(result == nullptr)) {
    std::unique_ptr<PredictionData> p(new PredictionData());
    PredictionData *previous = nullptr;
    if (__kmp_task_predictions_g.compare_exchange_strong(previous, p.get())) {
      result = p.release();
    } else {
      result = previous;
    }
  }

  return result;
}

__thread PredictionData *__kmp_task_prediction_data = NULL;

PredictionData &__kmp_get_prediction_data() {
  if (UNLIKELY(!__kmp_task_prediction_data)) {
    __kmp_task_prediction_data = __kmp_get_or_allocate_prediction_data();
  }
  return *__kmp_task_prediction_data;
}

enum prediction_info_masks
{
  PREDINFO_HAS_NO_PREDICTION = 0x1
};

__attribute__((used))
static const char *get_state_name(kmp_task_pred_state p) {
  switch (p) {
  default:
    return "<unknown state?>";
  case KTP_NONE:
    return "<none>";
  case KTP_READY:
    return "<ready>";
  case KTP_EXECUTE:
    return "<execute>";
  }
}

void __kmp_task_change_state(kmp_team_t* team,
                             kmp_task_pred_state old_state,
                             kmp_task_pred_state new_state, kmp_int32 label,
                             kmp_int32 cost,
                             kmp_int32* prediction_info) {
  PredictionData &pd = __kmp_get_prediction_data();

  kmp_int32 gtid = __kmp_get_gtid();
  pd.m.lock(gtid);
  std::pair<TaskTypeToPrediction::iterator, bool> it_insert =
      pd.task_type_to_pred.emplace(label, PredictionTaskType());
  PredictionTaskType &pred_tt = it_insert.first->second;
  pd.m.unlock(gtid);

  KA_TRACE(
      1,
      ("__kmp_task_change_state: label=%d cost=%d Changing state: %s -> %s\n",
       label, cost, get_state_name(old_state), get_state_name(new_state)));

  switch (old_state) {
  case KTP_READY:
    to_atomic(pred_tt.num_ready) -= 1;
    to_atomic(pred_tt.ready_wl) -= cost;
    break;
  case KTP_EXECUTE:
    to_atomic(pred_tt.exec_wl) -= cost;
    to_atomic(pred_tt.num_executing) -= 1;
    if (*prediction_info & PREDINFO_HAS_NO_PREDICTION) {
      KA_TRACE(
          1,
          ("__kmp_task_change_state: label=%d cost=%d Remove predictionless\n",
           label, cost));
      to_atomic(pred_tt.num_without_prediction) -= 1;
    }
    break;
  case KTP_NONE:
    break;
  default:
    KMP_DEBUG_ASSERT(false && "Invalid old state");
    break;
  }

  switch (new_state) {
  case KTP_READY:
    to_atomic(pred_tt.num_ready) += 1;
    to_atomic(pred_tt.ready_wl) += cost;
    if (label == kmp_default_task_label || pred_tt.cost_per_time.empty()) {
      KA_TRACE(
          1, ("__kmp_task_change_state: label=%d cost=%d Add predictionless\n",
              label, cost));
      to_atomic(pred_tt.num_without_prediction) += 1;
      (*prediction_info) |= PREDINFO_HAS_NO_PREDICTION;
    }
    to_atomic(team->t.last_prediction_timestamp).store(0);
    (void)__kmp_get_or_update_threads_prediction();
    break;
  case KTP_EXECUTE:
    to_atomic(pred_tt.exec_wl) += cost;
    to_atomic(pred_tt.num_executing) += 1;
    break;
  case KTP_NONE:
    break;
  default:
    KMP_DEBUG_ASSERT(false && "Invalid new state");
    break;
  }

}

// Global prediction data.

// Per thread prediction outcome.

bool __kmp_prediction_is_stale(kmp_uint64 time_of_prediction) {
  return (timespec_to_nanoseconds(__kmp_get_coarse_timespec()) -
          time_of_prediction) > TS_STALE_NS;
}

// "label" is a legacy term that actually means "kind of task" or "class type".
void __kmp_update_task_elapsed_time(kmp_int32 label, kmp_int32 cost,
                                    kmp_uint64 elapsed) {
  PredictionData &pd = __kmp_get_prediction_data();

  kmp_int32 gtid = __kmp_get_gtid();
  pd.m.lock(gtid);
  std::pair<TaskTypeToPrediction::iterator, bool> it_insert =
      pd.task_type_to_pred.emplace(label, PredictionTaskType());
  PredictionTaskType &pred_tt = it_insert.first->second;
  pd.m.unlock(gtid);

  // Use microseconds.
  elapsed = elapsed / 1000;

  double alpha = double(elapsed) / double(cost);
  KA_TRACE(1,
           ("__kmp_update_task_elapsed_time: label=%d cost=%d elapsed(µs)=%lu "
            "alpha=%e\n",
            label, cost, elapsed, alpha));
  pred_tt.cost_per_time.add(alpha);
}

enum Extrae_Event_Types
{
  extrae_prediction_type = 111,
};

enum Extrae_Event_Values
{
  extrae_query = 1,
  extrae_computing = 2,
  extrae_waking_up_thread = 3,
};

#if TASKPREDICTION_EXTRAE
extern "C" {
void Extrae_event(unsigned type, long long value) __attribute__((weak));
}

static inline void do_extrae_event(unsigned type, long long value) {
  if (Extrae_event) {
    Extrae_event(type, value);
  }
}
#else
static inline void do_extrae_event(unsigned, long long) { }
#endif

kmp_int32 __kmp_get_threads_prediction(PredictionData &pd,
                                       kmp_info_t *current_thread,
                                       kmp_team_t *team) {
  kmp_int32 threads_in_team = current_thread->th.th_team_nproc;

  // Use microseconds like the reference implementation does.
  kmp_int32 frequency = TS_STALE_NS / 1000;

  kmp_int32 result = 0;
  kmp_int32 num_tasks = 0;
  kmp_int32 gtid = __kmp_get_gtid();
  if (!pd.m.try_lock(gtid)) {
    // Return the old prediction if this is being computed at the moment.
    return to_atomic(team->t.predicted_threads).load();
  }
  do_extrae_event(extrae_prediction_type, extrae_computing);
  // For each task class.
  for (auto &tt_and_pred : pd.task_type_to_pred) {
    if (result >= threads_in_team)
      break;

    KA_TRACE(1, ("__kmp_get_threads_prediction: predicting for label=%d\n",
                 tt_and_pred.first));

    PredictionTaskType &pred_tt = tt_and_pred.second;
    result += to_atomic(pred_tt.num_without_prediction).load();

    double cost_per_time = pred_tt.cost_per_time.get_rolling_mean();
    double beta = (to_atomic(pred_tt.ready_wl) + to_atomic(pred_tt.exec_wl)) *
                  cost_per_time;
    KA_TRACE(1,
             ("__kmp_get_threads_prediction: predicting for result=%d label=%d "
              "cost_per_time=%e beta=%e\n",
              result, tt_and_pred.first, cost_per_time, beta));
    beta /= frequency;
    result += beta;

    KA_TRACE(1,
             ("__kmp_get_threads_prediction: predicting for result=%d label=%d "
              "cost_per_time=%e\n",
              result, tt_and_pred.first, cost_per_time));
    kmp_int32 num_ready = to_atomic(pred_tt.num_ready);
    kmp_int32 num_executing = to_atomic(pred_tt.num_executing);
    kmp_int32 num_tasks_type = num_ready + num_executing;
    KA_TRACE(1, ("__kmp_get_threads_prediction: predicting for label=%d "
                 "num_tasks=%d ready=%d executing=%d\n",
                 tt_and_pred.first, num_tasks_type, num_ready, num_executing));
    num_tasks += num_tasks_type;
  }
  pd.m.unlock(gtid);

  result = std::min(result, num_tasks);

  // Clamp the result to be in [2, numcpus]
  result = std::max(2, std::min(result, threads_in_team));
  // Add prediction to window of predictions.
  result = std::ceil(pd.thread_prediction.add(result));

  kmp_int32 old_predicted = to_atomic(team->t.predicted_threads).load();
  to_atomic(team->t.predicted_threads).store(result);
  to_atomic(team->t.last_prediction_timestamp)
      .store(timespec_to_nanoseconds(__kmp_get_coarse_timespec()));
  KA_TRACE(
      1, ("__kmp_get_threads_prediction: Number of threads predicted is: %d\n",
          result));

  if (result > old_predicted) {
    // Try to wakeup threads from this team if the prediction has increased.
    kmp_task_team_t *task_team = current_thread->th.th_task_team;
    if (task_team != NULL) {
      // Should we use th_team_nproc?
      kmp_int32 nthreads = task_team->tt.tt_nproc;

      kmp_thread_data_t *threads_data =
          (kmp_thread_data_t *)TCR_PTR(task_team->tt.tt_threads_data);
      if (threads_data) {
        KA_TRACE(1,
                 ("__kmp_get_threads_prediction: About to wake up threads\n"));
        kmp_int32 num_awaken = 0;
        for (int i = 0; i < nthreads; i++) {
          kmp_info_t *other_thread = threads_data[i].td.td_thr;
          if (other_thread != current_thread &&
              (TCR_PTR(CCAST(void *, other_thread->th.th_sleep_loc)) != NULL)) {
            KA_TRACE(
                1, ("__kmp_get_threads_prediction: Waking up thread#%d\n", i));
            do_extrae_event(extrae_prediction_type, extrae_waking_up_thread);
            __kmp_null_resume_wrapper(__kmp_gtid_from_thread(other_thread),
                                      other_thread->th.th_sleep_loc);
            do_extrae_event(extrae_prediction_type, 0);
          } else {
            KA_TRACE(
                1,
                ("__kmp_get_threads_prediction: Thread#%d is already awake\n",
                 i));
          }
          num_awaken++;
          if (num_awaken >= result)
            break;
        }
        KA_TRACE(
            1, ("__kmp_get_threads_prediction: Finished waking up threads\n"));
      } else {
        KA_TRACE(1, ("__kmp_get_threads_prediction: Cannot wake up threads "
                     "because there is no thread data\n"));
      }
    }
  }

  KA_TRACE(1, ("__kmp_get_threads_prediction: prediction %d\n", result));

  do_extrae_event(extrae_prediction_type, 0);
  return result;
}

kmp_int32 __kmp_get_or_update_threads_prediction() {
  do_extrae_event(extrae_prediction_type, extrae_query);

  kmp_int32 gtid = __kmp_get_gtid();
  kmp_info_t *current_thread = __kmp_threads[gtid];
  kmp_team_t *team = current_thread->th.th_team;

  if (!team) {
    do_extrae_event(extrae_prediction_type, 0);
    return -1;
  }

  PredictionData &pd = __kmp_get_prediction_data();
  kmp_int32 result;
  if (__kmp_prediction_is_stale(
          to_atomic(team->t.last_prediction_timestamp).load())) {
    result = __kmp_get_threads_prediction(pd, current_thread, team);
  } else {
    result = to_atomic(team->t.predicted_threads).load();
  }

  do_extrae_event(extrae_prediction_type, 0);
  return result;
}

kmp_int32 __kmp_predicted_threads() {
  kmp_int32 pred = __kmp_get_or_update_threads_prediction();
  return pred;
}

#ifdef LIBOMP_TASK_PREDICTION_TIMERS
void __kmp_prediction_start_timer() {
  PredictionData &pd = __kmp_get_prediction_data();
  pd.start_timer();
}

void __kmp_prediction_stop_timer() {
  PredictionData &pd = __kmp_get_prediction_data();
  pd.stop_timer();
}
#endif

extern "C" {

int __kmpc_get_predicted_threads(void) {
  return __kmp_predicted_threads();
}

void __kmpc_task_set_cost(kmp_task_t * task, kmp_int32 cost) {
  kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(task);
  taskdata->td_cost = cost;
}

void __kmpc_task_set_label(kmp_task_t *task, kmp_int32 label) {
  kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(task);
  taskdata->td_label = label;
}

}

#else

extern "C" {

void __kmpc_task_set_cost(kmp_task_t *, kmp_int32) {
  // KMP_WARNING(TaskPredictionIgnoringClause, "cost");
}

void __kmpc_task_set_label(kmp_task_t *, kmp_int32) {
  // KMP_WARNING(TaskPredictionIgnoringClause, "label");
}

}

#endif
