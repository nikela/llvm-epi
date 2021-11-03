/*
 * kmp_taskdeps.cpp
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//#define KMP_SUPPORT_GRAPH_OUTPUT 1

#include "kmp.h"
#include "kmp_io.h"
#include "kmp_wait_release.h"
#include "kmp_taskdeps.h"
#if OMPT_SUPPORT
#include "ompt-specific.h"
#endif

// TODO: Improve memory allocation? keep a list of pre-allocated structures?
// allocate in blocks? re-use list finished list entries?
// TODO: don't use atomic ref counters for stack-allocated nodes.
// TODO: find an alternate to atomic refs for heap-allocated nodes?
// TODO: Finish graph output support
// TODO: kmp_lock_t seems a tad to big (and heavy weight) for this. Check other
// runtime locks
// TODO: Any ITT support needed?

#if LIBOMP_TASKGRAPH
int recording;
int inside_taskgraph = false;
int taskify = true;
kmp_int32 ntdgs = 0;
kmp_int32 numRoots = 0;
kmp_int32 id_counter;

//Pointers
dynamic_tdg_info dynamic_tdgs[10];
kmp_record_info *RecordMap;
int *rootTasks;

//Sizes
kmp_int32 MaxNesting = 4;
kmp_int32 MapSize = 50;
kmp_int32 MapIncrement= 50;
kmp_int32 SuccessorsSize = 10;
kmp_int32 SuccessorsIncrement = 5;
kmp_int32 ColorMapSize = 20;

kmp_futex_lock_t taskgraph_lock;

// Colors for the graphviz output
const char* color_names[] = {
        "aquamarine3", "crimson", "chartreuse", "blue2", "darkorchid3", "darkgoldenrod1",
        "deeppink4", "gray19", "indigo", "indianred", "forestgreen", "navy", "orangered2",
        "slateblue3", "yellowgreen", "salmon", "purple", "mediumturquoise", "slategray3"
};

struct ident_color{
  const char * td_ident;
  const char *color;
};

ident_color *ColorMap;
kmp_int32 ColorIndex=0;
#endif // LIBOMP_TASKGRAPH

#ifdef KMP_SUPPORT_GRAPH_OUTPUT
static std::atomic<kmp_int32> kmp_node_id_seed = ATOMIC_VAR_INIT(0);
#endif

static void __kmp_init_node(kmp_depnode_t *node) {
  node->dn.successors = NULL;
  node->dn.task = NULL; // will point to the right task
  // once dependences have been processed
  for (int i = 0; i < MAX_MTX_DEPS; ++i)
    node->dn.mtx_locks[i] = NULL;
  node->dn.mtx_num_locks = 0;
  __kmp_init_lock(&node->dn.lock);
  KMP_ATOMIC_ST_RLX(&node->dn.nrefs, 1); // init creates the first reference
#ifdef KMP_SUPPORT_GRAPH_OUTPUT
  node->dn.id = KMP_ATOMIC_INC(&kmp_node_id_seed);
#endif
}

static inline kmp_depnode_t *__kmp_node_ref(kmp_depnode_t *node) {
  KMP_ATOMIC_INC(&node->dn.nrefs);
  return node;
}

enum { KMP_DEPHASH_OTHER_SIZE = 97, KMP_DEPHASH_MASTER_SIZE = 997 };

size_t sizes[] = {997, 2003, 4001, 8191, 16001, 32003, 64007, 131071, 270029};
const size_t MAX_GEN = 8;

static inline size_t __kmp_dephash_hash(kmp_intptr_t addr, size_t hsize) {
  // TODO alternate to try: set = (((Addr64)(addrUsefulBits * 9.618)) %
  // m_num_sets );
  return ((addr >> 6) ^ (addr >> 2)) % hsize;
}

static kmp_dephash_t *__kmp_dephash_extend(kmp_info_t *thread,
                                           kmp_dephash_t *current_dephash) {
  kmp_dephash_t *h;

  size_t gen = current_dephash->generation + 1;
  if (gen >= MAX_GEN)
    return current_dephash;
  size_t new_size = sizes[gen];

  size_t size_to_allocate =
      new_size * sizeof(kmp_dephash_entry_t *) + sizeof(kmp_dephash_t);

#if USE_FAST_MEMORY
  h = (kmp_dephash_t *)__kmp_fast_allocate(thread, size_to_allocate);
#else
  h = (kmp_dephash_t *)__kmp_thread_malloc(thread, size_to_allocate);
#endif

  h->size = new_size;
  h->nelements = current_dephash->nelements;
  h->buckets = (kmp_dephash_entry **)(h + 1);
  h->generation = gen;
  h->nconflicts = 0;

  // make sure buckets are properly initialized
  for (size_t i = 0; i < new_size; i++) {
    h->buckets[i] = NULL;
  }

  // insert existing elements in the new table
  for (size_t i = 0; i < current_dephash->size; i++) {
    kmp_dephash_entry_t *next, *entry;
    for (entry = current_dephash->buckets[i]; entry; entry = next) {
      next = entry->next_in_bucket;
      // Compute the new hash using the new size, and insert the entry in
      // the new bucket.
      size_t new_bucket = __kmp_dephash_hash(entry->addr, h->size);
      entry->next_in_bucket = h->buckets[new_bucket];
      if (entry->next_in_bucket) {
        h->nconflicts++;
      }
      h->buckets[new_bucket] = entry;
    }
  }

  // Free old hash table
#if USE_FAST_MEMORY
  __kmp_fast_free(thread, current_dephash);
#else
  __kmp_thread_free(thread, current_dephash);
#endif

  return h;
}

static kmp_dephash_t *__kmp_dephash_create(kmp_info_t *thread,
                                           kmp_taskdata_t *current_task) {
  kmp_dephash_t *h;

  size_t h_size;

  if (current_task->td_flags.tasktype == TASK_IMPLICIT)
    h_size = KMP_DEPHASH_MASTER_SIZE;
  else
    h_size = KMP_DEPHASH_OTHER_SIZE;

  size_t size = h_size * sizeof(kmp_dephash_entry_t *) + sizeof(kmp_dephash_t);

#if USE_FAST_MEMORY
  h = (kmp_dephash_t *)__kmp_fast_allocate(thread, size);
#else
  h = (kmp_dephash_t *)__kmp_thread_malloc(thread, size);
#endif
  h->size = h_size;

  h->generation = 0;
  h->nelements = 0;
  h->nconflicts = 0;
  h->buckets = (kmp_dephash_entry **)(h + 1);

  for (size_t i = 0; i < h_size; i++)
    h->buckets[i] = 0;

  return h;
}

static kmp_dephash_entry *__kmp_dephash_find(kmp_info_t *thread,
                                             kmp_dephash_t **hash,
                                             kmp_intptr_t addr) {
  kmp_dephash_t *h = *hash;
  if (h->nelements != 0 && h->nconflicts / h->size >= 1) {
    *hash = __kmp_dephash_extend(thread, h);
    h = *hash;
  }
  size_t bucket = __kmp_dephash_hash(addr, h->size);

  kmp_dephash_entry_t *entry;
  for (entry = h->buckets[bucket]; entry; entry = entry->next_in_bucket)
    if (entry->addr == addr)
      break;

  if (entry == NULL) {
// create entry. This is only done by one thread so no locking required
#if USE_FAST_MEMORY
    entry = (kmp_dephash_entry_t *)__kmp_fast_allocate(
        thread, sizeof(kmp_dephash_entry_t));
#else
    entry = (kmp_dephash_entry_t *)__kmp_thread_malloc(
        thread, sizeof(kmp_dephash_entry_t));
#endif
    entry->addr = addr;
    entry->last_out = NULL;
    entry->last_set = NULL;
    entry->prev_set = NULL;
    entry->last_flag = 0;
    entry->mtx_lock = NULL;
    entry->next_in_bucket = h->buckets[bucket];
    h->buckets[bucket] = entry;
    h->nelements++;
    if (entry->next_in_bucket)
      h->nconflicts++;
  }
  return entry;
}

static kmp_depnode_list_t *__kmp_add_node(kmp_info_t *thread,
                                          kmp_depnode_list_t *list,
                                          kmp_depnode_t *node) {
  kmp_depnode_list_t *new_head;

#if USE_FAST_MEMORY
  new_head = (kmp_depnode_list_t *)__kmp_fast_allocate(
      thread, sizeof(kmp_depnode_list_t));
#else
  new_head = (kmp_depnode_list_t *)__kmp_thread_malloc(
      thread, sizeof(kmp_depnode_list_t));
#endif

  new_head->node = __kmp_node_ref(node);
  new_head->next = list;

  return new_head;
}

static inline void __kmp_track_dependence(kmp_int32 gtid, kmp_depnode_t *source,
                                          kmp_depnode_t *sink,
                                          kmp_task_t *sink_task) {
#if LIBOMP_TASKGRAPH
  if (recording && inside_taskgraph) {
    kmp_record_info *SourceInfo = &(RecordMap[source->dn.part_id]);
    bool exists = false;
    for (int i = 0; i < SourceInfo->nsuccessors; i++) {
      if (SourceInfo->successors[i] == sink_task->part_id) {
        exists = true;
        break;
      }
    }
    if (!exists) {
      if (SourceInfo->nsuccessors >= SourceInfo->successorsSize) {
        SourceInfo->successorsSize += SuccessorsIncrement;
        SourceInfo->successors = (kmp_int32 *)realloc(
            SourceInfo->successors,
            SourceInfo->successorsSize * sizeof(kmp_int32));
      }

      SourceInfo->successors[SourceInfo->nsuccessors] = sink_task->part_id;
      SourceInfo->nsuccessors++;

      kmp_record_info *SinkInfo = &(RecordMap[sink_task->part_id]);
      SinkInfo->npredecessors++;
    }
  }
#endif

#ifdef KMP_SUPPORT_GRAPH_OUTPUT
  kmp_taskdata_t *task_source = KMP_TASK_TO_TASKDATA(source->dn.task);
  kmp_taskdata_t *task_sink = KMP_TASK_TO_TASKDATA(sink_task);

  // do not use sink->dn.task as that is only filled after the dependences
  // are already processed!

  __kmp_printf("%d(%s) -> %d(%s)\n", source->dn.id,
               task_source->td_ident->psource, sink->dn.id,
               task_sink->td_ident->psource);
#endif
#if OMPT_SUPPORT && OMPT_OPTIONAL
  /* OMPT tracks dependences between task (a=source, b=sink) in which
     task a blocks the execution of b through the ompt_new_dependence_callback
     */
  if (ompt_enabled.ompt_callback_task_dependence) {
    kmp_taskdata_t *task_source = KMP_TASK_TO_TASKDATA(source->dn.task);
    ompt_data_t *sink_data;
    if (sink_task)
      sink_data = &(KMP_TASK_TO_TASKDATA(sink_task)->ompt_task_info.task_data);
    else
      sink_data = &__kmp_threads[gtid]->th.ompt_thread_info.task_data;

    ompt_callbacks.ompt_callback(ompt_callback_task_dependence)(
        &(task_source->ompt_task_info.task_data), sink_data);
  }
#endif /* OMPT_SUPPORT && OMPT_OPTIONAL */
}

static inline kmp_int32
__kmp_depnode_link_successor(kmp_int32 gtid, kmp_info_t *thread,
                             kmp_task_t *task, kmp_depnode_t *node,
                             kmp_depnode_list_t *plist) {
  if (!plist)
    return 0;
  kmp_int32 npredecessors = 0;
  // link node as successor of list elements
  for (kmp_depnode_list_t *p = plist; p; p = p->next) {
    kmp_depnode_t *dep = p->node;
    __kmp_track_dependence(gtid, dep, node, task);
    if (dep->dn.task) {
      KMP_ACQUIRE_DEPNODE(gtid, dep);
      if (dep->dn.task) {
        //__kmp_track_dependence(gtid, dep, node, task);
        dep->dn.successors = __kmp_add_node(thread, dep->dn.successors, node);
        KA_TRACE(40, ("__kmp_process_deps: T#%d adding dependence from %p to "
                      "%p\n",
                      gtid, KMP_TASK_TO_TASKDATA(dep->dn.task),
                      KMP_TASK_TO_TASKDATA(task)));
        npredecessors++;
      }
      KMP_RELEASE_DEPNODE(gtid, dep);
    }
  }
  return npredecessors;
}

static inline kmp_int32 __kmp_depnode_link_successor(kmp_int32 gtid,
                                                     kmp_info_t *thread,
                                                     kmp_task_t *task,
                                                     kmp_depnode_t *source,
                                                     kmp_depnode_t *sink) {
  if (!sink)
    return 0;
  kmp_int32 npredecessors = 0;
  __kmp_track_dependence(gtid, sink, source, task);
  if (sink->dn.task) {
    // synchronously add source to sink' list of successors
    KMP_ACQUIRE_DEPNODE(gtid, sink);
    if (sink->dn.task) {
      //__kmp_track_dependence(gtid, sink, source, task);
      sink->dn.successors = __kmp_add_node(thread, sink->dn.successors, source);
      KA_TRACE(40, ("__kmp_process_deps: T#%d adding dependence from %p to "
                    "%p\n",
                    gtid, KMP_TASK_TO_TASKDATA(sink->dn.task),
                    KMP_TASK_TO_TASKDATA(task)));
      npredecessors++;
    }
    KMP_RELEASE_DEPNODE(gtid, sink);
  }
  return npredecessors;
}

template <bool filter>
static inline kmp_int32
__kmp_process_deps(kmp_int32 gtid, kmp_depnode_t *node, kmp_dephash_t **hash,
                   bool dep_barrier, kmp_int32 ndeps,
                   kmp_depend_info_t *dep_list, kmp_task_t *task) {
  KA_TRACE(30, ("__kmp_process_deps<%d>: T#%d processing %d dependences : "
                "dep_barrier = %d\n",
                filter, gtid, ndeps, dep_barrier));

  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_int32 npredecessors = 0;
  for (kmp_int32 i = 0; i < ndeps; i++) {
    const kmp_depend_info_t *dep = &dep_list[i];

    if (filter && dep->base_addr == 0)
      continue; // skip filtered entries

    kmp_dephash_entry_t *info =
        __kmp_dephash_find(thread, hash, dep->base_addr);
    kmp_depnode_t *last_out = info->last_out;
    kmp_depnode_list_t *last_set = info->last_set;
    kmp_depnode_list_t *prev_set = info->prev_set;

    if (dep->flags.out) { // out or inout --> clean lists if any
      if (last_set) {
        npredecessors +=
            __kmp_depnode_link_successor(gtid, thread, task, node, last_set);
        __kmp_depnode_list_free(thread, last_set);
        __kmp_depnode_list_free(thread, prev_set);
        info->last_set = NULL;
        info->prev_set = NULL;
        info->last_flag = 0; // no sets in this dephash entry
      } else {
        npredecessors +=
            __kmp_depnode_link_successor(gtid, thread, task, node, last_out);
      }
      __kmp_node_deref(thread, last_out);
      if (!dep_barrier) {
        info->last_out = __kmp_node_ref(node);
      } else {
        // if this is a sync point in the serial sequence, then the previous
        // outputs are guaranteed to be completed after the execution of this
        // task so the previous output nodes can be cleared.
        info->last_out = NULL;
      }
    } else { // either IN or MTX or SET
      if (info->last_flag == 0 || info->last_flag == dep->flag) {
        // last_set either didn't exist or of same dep kind
        // link node as successor of the last_out if any
        npredecessors +=
            __kmp_depnode_link_successor(gtid, thread, task, node, last_out);
        // link node as successor of all nodes in the prev_set if any
        npredecessors +=
            __kmp_depnode_link_successor(gtid, thread, task, node, prev_set);
        if (dep_barrier) {
          // clean last_out and prev_set if any; don't touch last_set
          __kmp_node_deref(thread, last_out);
          info->last_out = NULL;
          __kmp_depnode_list_free(thread, prev_set);
          info->prev_set = NULL;
        }
      } else { // last_set is of different dep kind, make it prev_set
        // link node as successor of all nodes in the last_set
        npredecessors +=
            __kmp_depnode_link_successor(gtid, thread, task, node, last_set);
        // clean last_out if any
        __kmp_node_deref(thread, last_out);
        info->last_out = NULL;
        // clean prev_set if any
        __kmp_depnode_list_free(thread, prev_set);
        if (!dep_barrier) {
          // move last_set to prev_set, new last_set will be allocated
          info->prev_set = last_set;
        } else {
          info->prev_set = NULL;
          info->last_flag = 0;
        }
        info->last_set = NULL;
      }
      // for dep_barrier last_flag value should remain:
      // 0 if last_set is empty, unchanged otherwise
      if (!dep_barrier) {
        info->last_flag = dep->flag; // store dep kind of the last_set
        info->last_set = __kmp_add_node(thread, info->last_set, node);
      }
      // check if we are processing MTX dependency
      if (dep->flag == KMP_DEP_MTX) {
        if (info->mtx_lock == NULL) {
          info->mtx_lock = (kmp_lock_t *)__kmp_allocate(sizeof(kmp_lock_t));
          __kmp_init_lock(info->mtx_lock);
        }
        KMP_DEBUG_ASSERT(node->dn.mtx_num_locks < MAX_MTX_DEPS);
        kmp_int32 m;
        // Save lock in node's array
        for (m = 0; m < MAX_MTX_DEPS; ++m) {
          // sort pointers in decreasing order to avoid potential livelock
          if (node->dn.mtx_locks[m] < info->mtx_lock) {
            KMP_DEBUG_ASSERT(!node->dn.mtx_locks[node->dn.mtx_num_locks]);
            for (int n = node->dn.mtx_num_locks; n > m; --n) {
              // shift right all lesser non-NULL pointers
              KMP_DEBUG_ASSERT(node->dn.mtx_locks[n - 1] != NULL);
              node->dn.mtx_locks[n] = node->dn.mtx_locks[n - 1];
            }
            node->dn.mtx_locks[m] = info->mtx_lock;
            break;
          }
        }
        KMP_DEBUG_ASSERT(m < MAX_MTX_DEPS); // must break from loop
        node->dn.mtx_num_locks++;
      }
    }
  }
  KA_TRACE(30, ("__kmp_process_deps<%d>: T#%d found %d predecessors\n", filter,
                gtid, npredecessors));
  return npredecessors;
}

#define NO_DEP_BARRIER (false)
#define DEP_BARRIER (true)

// returns true if the task has any outstanding dependence
static bool __kmp_check_deps(kmp_int32 gtid, kmp_depnode_t *node,
                             kmp_task_t *task, kmp_dephash_t **hash,
                             bool dep_barrier, kmp_int32 ndeps,
                             kmp_depend_info_t *dep_list,
                             kmp_int32 ndeps_noalias,
                             kmp_depend_info_t *noalias_dep_list) {
  int i, n_mtxs = 0;
#if KMP_DEBUG
  kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(task);
#endif
  KA_TRACE(20, ("__kmp_check_deps: T#%d checking dependences for task %p : %d "
                "possibly aliased dependences, %d non-aliased dependences : "
                "dep_barrier=%d .\n",
                gtid, taskdata, ndeps, ndeps_noalias, dep_barrier));

  // Filter deps in dep_list
  // TODO: Different algorithm for large dep_list ( > 10 ? )
  for (i = 0; i < ndeps; i++) {
    if (dep_list[i].base_addr != 0) {
      KMP_DEBUG_ASSERT(
          dep_list[i].flag == KMP_DEP_IN || dep_list[i].flag == KMP_DEP_OUT ||
          dep_list[i].flag == KMP_DEP_INOUT ||
          dep_list[i].flag == KMP_DEP_MTX || dep_list[i].flag == KMP_DEP_SET);
      for (int j = i + 1; j < ndeps; j++) {
        if (dep_list[i].base_addr == dep_list[j].base_addr) {
          if (dep_list[i].flag != dep_list[j].flag) {
            // two different dependences on same address work identical to OUT
            dep_list[i].flag = KMP_DEP_OUT;
          }
          dep_list[j].base_addr = 0; // Mark j element as void
        }
      }
      if (dep_list[i].flag == KMP_DEP_MTX) {
        // limit number of mtx deps to MAX_MTX_DEPS per node
        if (n_mtxs < MAX_MTX_DEPS && task != NULL) {
          ++n_mtxs;
        } else {
          dep_list[i].flag = KMP_DEP_OUT; // downgrade mutexinoutset to inout
        }
      }
    }
  }

  // doesn't need to be atomic as no other thread is going to be accessing this
  // node just yet.
  // npredecessors is set -1 to ensure that none of the releasing tasks queues
  // this task before we have finished processing all the dependences
  node->dn.npredecessors = -1;

  // used to pack all npredecessors additions into a single atomic operation at
  // the end
  int npredecessors;

  npredecessors = __kmp_process_deps<true>(gtid, node, hash, dep_barrier, ndeps,
                                           dep_list, task);
  npredecessors += __kmp_process_deps<false>(
      gtid, node, hash, dep_barrier, ndeps_noalias, noalias_dep_list, task);

  node->dn.task = task;
  KMP_MB();

  // Account for our initial fake value
  npredecessors++;

  // Update predecessors and obtain current value to check if there are still
  // any outstanding dependences (some tasks may have finished while we
  // processed the dependences)
  npredecessors =
      node->dn.npredecessors.fetch_add(npredecessors) + npredecessors;

  KA_TRACE(20, ("__kmp_check_deps: T#%d found %d predecessors for task %p \n",
                gtid, npredecessors, taskdata));

  // beyond this point the task could be queued (and executed) by a releasing
  // task...
  return npredecessors > 0 ? true : false;
}

/*!
@ingroup TASKING
@param loc_ref location of the original task directive
@param gtid Global Thread ID of encountering thread
@param new_task task thunk allocated by __kmp_omp_task_alloc() for the ''new
task''
@param ndeps Number of depend items with possible aliasing
@param dep_list List of depend items with possible aliasing
@param ndeps_noalias Number of depend items with no aliasing
@param noalias_dep_list List of depend items with no aliasing

@return Returns either TASK_CURRENT_NOT_QUEUED if the current task was not
suspended and queued, or TASK_CURRENT_QUEUED if it was suspended and queued

Schedule a non-thread-switchable task with dependences for execution
*/
kmp_int32 __kmpc_omp_task_with_deps(ident_t *loc_ref, kmp_int32 gtid,
                                    kmp_task_t *new_task, kmp_int32 ndeps,
                                    kmp_depend_info_t *dep_list,
                                    kmp_int32 ndeps_noalias,
                                    kmp_depend_info_t *noalias_dep_list) {

  kmp_taskdata_t *new_taskdata = KMP_TASK_TO_TASKDATA(new_task);
  KA_TRACE(10, ("__kmpc_omp_task_with_deps(enter): T#%d loc=%p task=%p\n", gtid,
                loc_ref, new_taskdata));
  __kmp_assert_valid_gtid(gtid);
  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_taskdata_t *current_task = thread->th.th_current_task;

#if LIBOMP_TASKGRAPH
  if (recording && inside_taskgraph) {
    // Extend Map Size if needed
    if (new_task->part_id >= MapSize) {

      int OldSize = MapSize;
      MapSize += MapIncrement;
      RecordMap = (kmp_record_info *)realloc(RecordMap,
                                             MapSize * sizeof(kmp_record_info));

      for (int i = OldSize; i < MapSize; i++) {
        kmp_int32 *successorsList =
            (kmp_int32 *)malloc(SuccessorsSize * sizeof(kmp_int32));
        kmp_record_info newRecord = {successorsList, 0, nullptr, 0, 0, nullptr,
                                     SuccessorsSize};
        RecordMap[i] = newRecord;
      }
    }

    RecordMap[new_task->part_id].td_ident = new_taskdata->td_ident->psource;
    RecordMap[new_task->part_id].static_id = new_taskdata->td_task_id;
    RecordMap[new_task->part_id].task = new_task;
  }

  if (!recording && inside_taskgraph) {
    // TODO: Measure lock impact, maybe is better a lock per node
    __kmp_acquire_futex_lock(&taskgraph_lock, gtid);

    kmp_record_info *TaskInfo = &(RecordMap[new_task->part_id]);
    TaskInfo->task = new_task;

    if (!TaskInfo->npredecessors_counter) {
      // printf("  [OpenMP] Task Found  %d -> Executing, dependencies solved\n",
      // new_task->part_id);
      int result = __kmp_omp_task(gtid, new_task, true);
      __kmp_release_futex_lock(&taskgraph_lock, 0);
      return result;

    } else {
      // printf("  [OpenMP] Task Found %d deps:%d-> Waiting, dependencies not
      // solved \n", new_task->part_id, TaskInfo->npredecessors_counter);
      __kmp_release_futex_lock(&taskgraph_lock, 0);
      return TASK_CURRENT_NOT_QUEUED;
    }
  }
#endif

#if OMPT_SUPPORT
  if (ompt_enabled.enabled) {
    if (!current_task->ompt_task_info.frame.enter_frame.ptr)
      current_task->ompt_task_info.frame.enter_frame.ptr =
          OMPT_GET_FRAME_ADDRESS(0);
    if (ompt_enabled.ompt_callback_task_create) {
      ompt_callbacks.ompt_callback(ompt_callback_task_create)(
          &(current_task->ompt_task_info.task_data),
          &(current_task->ompt_task_info.frame),
          &(new_taskdata->ompt_task_info.task_data),
          ompt_task_explicit | TASK_TYPE_DETAILS_FORMAT(new_taskdata), 1,
          OMPT_LOAD_OR_GET_RETURN_ADDRESS(gtid));
    }

    new_taskdata->ompt_task_info.frame.enter_frame.ptr =
        OMPT_GET_FRAME_ADDRESS(0);
  }

#if OMPT_OPTIONAL
  /* OMPT grab all dependences if requested by the tool */
  if (ndeps + ndeps_noalias > 0 && ompt_enabled.ompt_callback_dependences) {
    kmp_int32 i;

    int ompt_ndeps = ndeps + ndeps_noalias;
    ompt_dependence_t *ompt_deps = (ompt_dependence_t *)KMP_OMPT_DEPS_ALLOC(
        thread, (ndeps + ndeps_noalias) * sizeof(ompt_dependence_t));

    KMP_ASSERT(ompt_deps != NULL);

    for (i = 0; i < ndeps; i++) {
      ompt_deps[i].variable.ptr = (void *)dep_list[i].base_addr;
      if (dep_list[i].flags.in && dep_list[i].flags.out)
        ompt_deps[i].dependence_type = ompt_dependence_type_inout;
      else if (dep_list[i].flags.out)
        ompt_deps[i].dependence_type = ompt_dependence_type_out;
      else if (dep_list[i].flags.in)
        ompt_deps[i].dependence_type = ompt_dependence_type_in;
      else if (dep_list[i].flags.mtx)
        ompt_deps[i].dependence_type = ompt_dependence_type_mutexinoutset;
      else if (dep_list[i].flags.set)
        ompt_deps[i].dependence_type = ompt_dependence_type_inoutset;
    }
    for (i = 0; i < ndeps_noalias; i++) {
      ompt_deps[ndeps + i].variable.ptr = (void *)noalias_dep_list[i].base_addr;
      if (noalias_dep_list[i].flags.in && noalias_dep_list[i].flags.out)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_inout;
      else if (noalias_dep_list[i].flags.out)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_out;
      else if (noalias_dep_list[i].flags.in)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_in;
      else if (noalias_dep_list[i].flags.mtx)
        ompt_deps[ndeps + i].dependence_type =
            ompt_dependence_type_mutexinoutset;
      else if (noalias_dep_list[i].flags.set)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_inoutset;
    }
    ompt_callbacks.ompt_callback(ompt_callback_dependences)(
        &(new_taskdata->ompt_task_info.task_data), ompt_deps, ompt_ndeps);
    /* We can now free the allocated memory for the dependences */
    /* For OMPD we might want to delay the free until end of this function */
    KMP_OMPT_DEPS_FREE(thread, ompt_deps);
  }
#endif /* OMPT_OPTIONAL */
#endif /* OMPT_SUPPORT */

  bool serial = current_task->td_flags.team_serial ||
                current_task->td_flags.tasking_ser ||
                current_task->td_flags.final;
  kmp_task_team_t *task_team = thread->th.th_task_team;
  serial = serial &&
           !(task_team && (task_team->tt.tt_found_proxy_tasks ||
                           task_team->tt.tt_hidden_helper_task_encountered));

  if (!serial && (ndeps > 0 || ndeps_noalias > 0)) {
    /* if no dependences have been tracked yet, create the dependence hash */
    if (current_task->td_dephash == NULL)
      current_task->td_dephash = __kmp_dephash_create(thread, current_task);

#if USE_FAST_MEMORY
    kmp_depnode_t *node =
        (kmp_depnode_t *)__kmp_fast_allocate(thread, sizeof(kmp_depnode_t));
#else
    kmp_depnode_t *node =
        (kmp_depnode_t *)__kmp_thread_malloc(thread, sizeof(kmp_depnode_t));
#endif

    __kmp_init_node(node);
#if LIBOMP_TASKGRAPH
    node->dn.part_id= new_task->part_id;
#endif
    new_taskdata->td_depnode = node;

    if (__kmp_check_deps(gtid, node, new_task, &current_task->td_dephash,
                         NO_DEP_BARRIER, ndeps, dep_list, ndeps_noalias,
                         noalias_dep_list)) {
      KA_TRACE(10, ("__kmpc_omp_task_with_deps(exit): T#%d task had blocking "
                    "dependences: "
                    "loc=%p task=%p, return: TASK_CURRENT_NOT_QUEUED\n",
                    gtid, loc_ref, new_taskdata));
#if OMPT_SUPPORT
      if (ompt_enabled.enabled) {
        current_task->ompt_task_info.frame.enter_frame = ompt_data_none;
      }
#endif
      return TASK_CURRENT_NOT_QUEUED;
    }
  } else {
    KA_TRACE(10, ("__kmpc_omp_task_with_deps(exit): T#%d ignored dependences "
                  "for task (serialized) loc=%p task=%p\n",
                  gtid, loc_ref, new_taskdata));
  }

  KA_TRACE(10, ("__kmpc_omp_task_with_deps(exit): T#%d task had no blocking "
                "dependences : "
                "loc=%p task=%p, transferring to __kmp_omp_task\n",
                gtid, loc_ref, new_taskdata));

  kmp_int32 ret = __kmp_omp_task(gtid, new_task, true);
#if OMPT_SUPPORT
  if (ompt_enabled.enabled) {
    current_task->ompt_task_info.frame.enter_frame = ompt_data_none;
  }
#endif
  return ret;
}

#if OMPT_SUPPORT
void __ompt_taskwait_dep_finish(kmp_taskdata_t *current_task,
                                ompt_data_t *taskwait_task_data) {
  if (ompt_enabled.ompt_callback_task_schedule) {
    ompt_callbacks.ompt_callback(ompt_callback_task_schedule)(
        taskwait_task_data, ompt_taskwait_complete, NULL);
  }
  current_task->ompt_task_info.frame.enter_frame.ptr = NULL;
  *taskwait_task_data = ompt_data_none;
}
#endif /* OMPT_SUPPORT */

#if LIBOMP_TASKGRAPH
void print_tdg() {
  for (int i = 1; i < MapSize; i++) {
    if (RecordMap[i].td_ident == nullptr)
      break;
    printf("TASK: %d Successors: ", RecordMap[i].static_id);
    for (int j = 0; j < RecordMap[i].nsuccessors; j++) {
      printf(" %d ", RecordMap[RecordMap[i].successors[j]].static_id);
    }
    printf(" Predecessors : %d ", RecordMap[i].npredecessors);
    printf(" \n");
  }
}

// Depth First Search to look for transitive edges
void traverse_node(kmp_int32 *edges_to_check, kmp_int32 *num_edges,
                   kmp_int32 node, kmp_int32 nesting_level, int Visited[]) {
  kmp_int32 *successors = RecordMap[node].successors;
  kmp_int32 nsuccessors = RecordMap[node].nsuccessors;
  Visited[node] = true;
  for (int i = 0; i < nsuccessors; i++) {
    kmp_int32 successor = successors[i];
    for (int j = 0; j < *num_edges; j++) {
      kmp_int32 edge = edges_to_check[j];
      if (edge == successor) {
        // Remove edge
        edges_to_check[j] = -1;
        for (int x = j; x < (*num_edges) - 1; x++) {
          edges_to_check[x] = edges_to_check[x + 1];
          edges_to_check[x + 1] = -1;
        }
        *num_edges = *num_edges - 1;
        RecordMap[edge].npredecessors--;
        break;
      }
    }
    if (Visited[successor] == false && nesting_level < MaxNesting)
      traverse_node(edges_to_check, num_edges, successor, nesting_level + 1,
                    Visited);
  }
}

void erase_transitive_edges() {
  for (int i = 1; i < MapSize; i++) {

    if (RecordMap[i].td_ident == nullptr)
      break;
    kmp_int32 nsuccessors = RecordMap[i].nsuccessors;

    if (!nsuccessors)
      continue;

    int Visited[MapSize];
    memset(Visited, false, sizeof(int) * MapSize);
    Visited[i] = true;
    // Copy succesors, as they may be modified
    kmp_int32 *successors =
        (kmp_int32 *)malloc(sizeof(kmp_int32) * nsuccessors);
    memcpy(successors, RecordMap[i].successors,
           sizeof(kmp_int32) * nsuccessors);

    for (int j = 0; j < nsuccessors; j++) {
      bool deleted = true;
      for (int x = 0; x < nsuccessors; x++) {
        if (RecordMap[i].successors[x] == successors[j])
          deleted = false;
      }
      if (!deleted)
        traverse_node(RecordMap[i].successors, &RecordMap[i].nsuccessors,
                      successors[j], 0, Visited);
    }
    // free succesors
    free(successors);
  }
}

void print_tdg_to_dot(void) {

  FILE *f = fopen("tdg.dot", "w");

  if (f == NULL) {
    printf("Error opening file!\n");
    exit(1);
  }

  fprintf(f, "digraph TDG {\n");
  fprintf(f, "   compound=true\n");
  fprintf(f, "   subgraph cluster_0 {\n");
  fprintf(f, "      label=TDG_%d\n", ntdgs);

  for (int i = 1; i < MapSize; i++) {

    if (RecordMap[i].td_ident == nullptr)
      break;
    const char *color = nullptr;
    const char *ident = RecordMap[i].td_ident;
    for (int j = 0; j < ColorMapSize; j++) {

      if (ColorMap[j].td_ident == nullptr) {
        ColorMap[j].td_ident = ident;
        ColorMap[j].color = color_names[ColorIndex];
        ColorIndex++;
        color = ColorMap[j].color;
        break;
      } else if (ColorMap[j].td_ident == ident) {
        color = ColorMap[j].color;
        break;
      }
    }

    if (color == nullptr) {
      printf("Unexpected error, color not found \n");
    } else {
      fprintf(f, "      %d[color=%s,style=bold]\n", RecordMap[i].static_id,
              color);
    }
  }
  fprintf(f, "   }\n");

  for (int i = 1; i < MapSize; i++) {

    if (RecordMap[i].td_ident == nullptr)
      break;

    kmp_int32 nsuccessors = RecordMap[i].nsuccessors;
    kmp_int32 *successors = RecordMap[i].successors;
    if (nsuccessors) {
      for (int j = 0; j < nsuccessors; j++) {
        fprintf(f, "   %d -> %d \n", RecordMap[i].static_id,
                RecordMap[successors[j]].static_id);
      }
    } else {
      fprintf(f, "   %d \n", RecordMap[i].static_id);
    }
  }

  fprintf(f, "   node [shape=plaintext];\n");
  fprintf(f, "    subgraph cluster_1000 {\n");
  fprintf(f, "      label=\"User functions:\"; style=\"rounded\";\n");
  fprintf(f, " user_funcs [label=<<table border=\"0\" cellspacing=\"10\" "
             "cellborder=\"0\">\n");
  for (int i = 0; i < ColorMapSize; i++) {
    if (ColorMap[i].td_ident == nullptr)
      break;
    fprintf(f, "      <tr>\n");
    fprintf(f,
            "         <td bgcolor=\"%s\" width=\"15px\" border=\"1\"></td>\n",
            ColorMap[i].color);
    fprintf(f, "         <td>%s</td>\n", ColorMap[i].td_ident);
    fprintf(f, "      </tr>\n");
  }
  fprintf(f, "      </table>>]\n");
  fprintf(f, "}}\n");
  fclose(f);
}

kmp_int32 __kmpc_taskgraph(ident_t *loc_ref, kmp_int32 gtid,
                               void (*entry)(void *), void *args,
                               kmp_uint32 condition) {
  recording = true;
  inside_taskgraph = true;
  id_counter = 0;
  __kmp_init_futex_lock(&taskgraph_lock);
  if (condition) {
    printf("[OpenMP] Condition true, forcing rerecording \n");
    // Implement the free of the previous RecordMap
  } else {
    for (int i = 0; i < ntdgs; i++) {
      if (dynamic_tdgs[i].loc == loc_ref->psource) {
        recording = false;
        RecordMap = dynamic_tdgs[i].RecordMap;
        if (taskify)
          for (int i = 1; i < MapSize; i++) {

            if (RecordMap[i].td_ident == nullptr)
              break;

            kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(RecordMap[i].task);

            KMP_ATOMIC_INC(&taskdata->td_parent->td_incomplete_child_tasks);

            // Protect with if?
            if (taskdata->td_parent->td_taskgroup)
              KMP_ATOMIC_INC(&taskdata->td_parent->td_taskgroup->count);

            if (taskdata->td_parent->td_flags.tasktype == TASK_EXPLICIT)
              KMP_ATOMIC_INC(&taskdata->td_parent->td_allocated_child_tasks);
          }
        break;
      }
    }
  }
  if (recording) {

    char *var = getenv("OMP_TASKIFY");
    if ((var && strcmp(var, "FALSE") == 0)) {
      taskify = false;
      // printf("[OpenMP] Tasksgraphs are not taskifyed \n");
    } else {
      // printf("[OpenMP] Taskgraphs are taskifyed \n");
    }
    RecordMap = (kmp_record_info *)malloc(MapSize * sizeof(kmp_record_info));
    ColorMap = (ident_color *)malloc(ColorMapSize * sizeof(ident_color));

    for (int i = 0; i < MapSize; i++) {
      kmp_int32 *successorsList =
          (kmp_int32 *)malloc(SuccessorsSize * sizeof(kmp_int32));
      kmp_record_info newRecord = {successorsList, 0, nullptr,       0, 0,
                                   nullptr,        0, SuccessorsSize};
      RecordMap[i] = newRecord;
    }
    for (int i = 0; i < ColorMapSize; i++) {
      ColorMap[i] = {nullptr, nullptr};
    }
    // printf("[OpenMP] Finish initializing Record map \n");
  }

  if (recording || !taskify) {
    entry(args);
  } else {
    // Only for taskified functions
    for (int i = 0; i < numRoots; i++) {
      __kmp_omp_task(gtid, RecordMap[rootTasks[i]].task, true);
    }
  }

  if (recording) {

    // Store roots
    rootTasks = (kmp_int32 *)malloc(MapSize * sizeof(kmp_int32));
    for (int i = 0; i < MapSize; i++) {
      if (RecordMap[i].td_ident != nullptr && RecordMap[i].npredecessors == 0) {
        rootTasks[numRoots] = i;
        numRoots++;
      }
    }
    dynamic_tdgs[ntdgs] = {loc_ref->psource, RecordMap};
    ntdgs++;
    // printf("[OpenMP] Recording finished! \n");
    erase_transitive_edges();
    // print_tdg();
    char *my_env_var = getenv("OMP_PRINT_TDG");
    if (my_env_var && strcmp(my_env_var, "TRUE") == 0) {
      // printf("[OpenMP] Dot file tdg.dot generated \n");
      print_tdg_to_dot();
    }
  }

  inside_taskgraph = false;

  __kmpc_omp_taskwait(loc_ref, gtid);

  if (recording)
    for (int i = 0; i < MapSize; i++) {
      if (RecordMap[i].td_ident != nullptr)
        RecordMap[i].npredecessors_counter = RecordMap[i].npredecessors;
    }

  return 1;
}
#endif

/*!
@ingroup TASKING
@param loc_ref location of the original task directive
@param gtid Global Thread ID of encountering thread
@param ndeps Number of depend items with possible aliasing
@param dep_list List of depend items with possible aliasing
@param ndeps_noalias Number of depend items with no aliasing
@param noalias_dep_list List of depend items with no aliasing

Blocks the current task until all specifies dependences have been fulfilled.
*/
void __kmpc_omp_wait_deps(ident_t *loc_ref, kmp_int32 gtid, kmp_int32 ndeps,
                          kmp_depend_info_t *dep_list, kmp_int32 ndeps_noalias,
                          kmp_depend_info_t *noalias_dep_list) {
  KA_TRACE(10, ("__kmpc_omp_wait_deps(enter): T#%d loc=%p\n", gtid, loc_ref));

  if (ndeps == 0 && ndeps_noalias == 0) {
    KA_TRACE(10, ("__kmpc_omp_wait_deps(exit): T#%d has no dependences to "
                  "wait upon : loc=%p\n",
                  gtid, loc_ref));
    return;
  }
  __kmp_assert_valid_gtid(gtid);
  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_taskdata_t *current_task = thread->th.th_current_task;

#if OMPT_SUPPORT
  // this function represents a taskwait construct with depend clause
  // We signal 4 events:
  //  - creation of the taskwait task
  //  - dependences of the taskwait task
  //  - schedule and finish of the taskwait task
  ompt_data_t *taskwait_task_data = &thread->th.ompt_thread_info.task_data;
  KMP_ASSERT(taskwait_task_data->ptr == NULL);
  if (ompt_enabled.enabled) {
    if (!current_task->ompt_task_info.frame.enter_frame.ptr)
      current_task->ompt_task_info.frame.enter_frame.ptr =
          OMPT_GET_FRAME_ADDRESS(0);
    if (ompt_enabled.ompt_callback_task_create) {
      ompt_callbacks.ompt_callback(ompt_callback_task_create)(
          &(current_task->ompt_task_info.task_data),
          &(current_task->ompt_task_info.frame), taskwait_task_data,
          ompt_task_taskwait | ompt_task_undeferred | ompt_task_mergeable, 1,
          OMPT_LOAD_OR_GET_RETURN_ADDRESS(gtid));
    }
  }

#if OMPT_OPTIONAL
  /* OMPT grab all dependences if requested by the tool */
  if (ndeps + ndeps_noalias > 0 && ompt_enabled.ompt_callback_dependences) {
    kmp_int32 i;

    int ompt_ndeps = ndeps + ndeps_noalias;
    ompt_dependence_t *ompt_deps = (ompt_dependence_t *)KMP_OMPT_DEPS_ALLOC(
        thread, (ndeps + ndeps_noalias) * sizeof(ompt_dependence_t));

    KMP_ASSERT(ompt_deps != NULL);

    for (i = 0; i < ndeps; i++) {
      ompt_deps[i].variable.ptr = (void *)dep_list[i].base_addr;
      if (dep_list[i].flags.in && dep_list[i].flags.out)
        ompt_deps[i].dependence_type = ompt_dependence_type_inout;
      else if (dep_list[i].flags.out)
        ompt_deps[i].dependence_type = ompt_dependence_type_out;
      else if (dep_list[i].flags.in)
        ompt_deps[i].dependence_type = ompt_dependence_type_in;
      else if (dep_list[i].flags.mtx)
        ompt_deps[ndeps + i].dependence_type =
            ompt_dependence_type_mutexinoutset;
      else if (dep_list[i].flags.set)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_inoutset;
    }
    for (i = 0; i < ndeps_noalias; i++) {
      ompt_deps[ndeps + i].variable.ptr = (void *)noalias_dep_list[i].base_addr;
      if (noalias_dep_list[i].flags.in && noalias_dep_list[i].flags.out)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_inout;
      else if (noalias_dep_list[i].flags.out)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_out;
      else if (noalias_dep_list[i].flags.in)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_in;
      else if (noalias_dep_list[i].flags.mtx)
        ompt_deps[ndeps + i].dependence_type =
            ompt_dependence_type_mutexinoutset;
      else if (noalias_dep_list[i].flags.set)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_inoutset;
    }
    ompt_callbacks.ompt_callback(ompt_callback_dependences)(
        taskwait_task_data, ompt_deps, ompt_ndeps);
    /* We can now free the allocated memory for the dependences */
    /* For OMPD we might want to delay the free until end of this function */
    KMP_OMPT_DEPS_FREE(thread, ompt_deps);
    ompt_deps = NULL;
  }
#endif /* OMPT_OPTIONAL */
#endif /* OMPT_SUPPORT */

  // We can return immediately as:
  // - dependences are not computed in serial teams (except with proxy tasks)
  // - if the dephash is not yet created it means we have nothing to wait for
  bool ignore = current_task->td_flags.team_serial ||
                current_task->td_flags.tasking_ser ||
                current_task->td_flags.final;
  ignore = ignore && thread->th.th_task_team != NULL &&
           thread->th.th_task_team->tt.tt_found_proxy_tasks == FALSE;
  ignore = ignore || current_task->td_dephash == NULL;

  if (ignore) {
    KA_TRACE(10, ("__kmpc_omp_wait_deps(exit): T#%d has no blocking "
                  "dependences : loc=%p\n",
                  gtid, loc_ref));
#if OMPT_SUPPORT
    __ompt_taskwait_dep_finish(current_task, taskwait_task_data);
#endif /* OMPT_SUPPORT */
    return;
  }

  kmp_depnode_t node = {0};
  __kmp_init_node(&node);

  if (!__kmp_check_deps(gtid, &node, NULL, &current_task->td_dephash,
                        DEP_BARRIER, ndeps, dep_list, ndeps_noalias,
                        noalias_dep_list)) {
    KA_TRACE(10, ("__kmpc_omp_wait_deps(exit): T#%d has no blocking "
                  "dependences : loc=%p\n",
                  gtid, loc_ref));
#if OMPT_SUPPORT
    __ompt_taskwait_dep_finish(current_task, taskwait_task_data);
#endif /* OMPT_SUPPORT */
    return;
  }

  int thread_finished = FALSE;
  kmp_flag_32<false, false> flag(
      (std::atomic<kmp_uint32> *)&node.dn.npredecessors, 0U);
  while (node.dn.npredecessors > 0) {
    flag.execute_tasks(thread, gtid, FALSE,
                       &thread_finished USE_ITT_BUILD_ARG(NULL),
                       __kmp_task_stealing_constraint);
  }

#if OMPT_SUPPORT
  __ompt_taskwait_dep_finish(current_task, taskwait_task_data);
#endif /* OMPT_SUPPORT */
  KA_TRACE(10, ("__kmpc_omp_wait_deps(exit): T#%d finished waiting : loc=%p\n",
                gtid, loc_ref));
}
