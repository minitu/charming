//#include "scheduler.h"
#include "charming.h"
#include "chare.h"
#include "ringbuf.h"
#include "util.h"

using namespace charm;

extern __device__ spsc_ringbuf_t* mbuf;
extern __device__ size_t mbuf_size;

extern __device__ chare_proxy_base* chare_proxies[];

__device__ envelope* charm::create_envelope(msgtype type, size_t msg_size) {
  /*
  // Secure region in my message pool
  ringbuf_off_t mret = spsc_ringbuf_acquire(mbuf, msg_size);
  assert(mret != -1 && mret < mbuf_size);

  // Create envelope
  envelope* env = new (mbuf->addr(mret)) envelope(type, msg_size, nvshmem_my_pe());

  return env;
  */
}

__device__ void charm::send_msg(envelope* env, size_t msg_size, int dst_pe) {
  /*
  int my_pe = nvshmem_my_pe();
  spsc_ringbuf_produce(mbuf);

  // TODO: Don't assume no wrap-around, use watermark
  // Retrieve symmetric addresses for metadata
  MsgQueueMeta* send_meta = send_meta_shell[dst_pe].meta;
  MsgQueueMeta* recv_meta = recv_meta_shell[my_pe].meta;
  offset_t avail = msg_queue_size - send_meta->write;
  assert(avail >= msg_size);

  // Send message
  MsgQueueShell& msgq_shell = msg_queue_shell[my_pe];
  nvshmem_char_put((char*)msgq_shell.addr(send_meta->write), (char*)env, env->size, dst_pe);
  send_meta->write += msg_size;

  // FIXME: Don't update receiver's write offset every time?
  nvshmem_fence();
  nvshmem_longlong_atomic_set(&recv_meta->write, send_meta->write, dst_pe);

  // Free region in my message pool
  size_t len, off;
  len = spsc_ringbuf_consume(mbuf, &off);
  spsc_ringbuf_release(mbuf, len);
  */
}

__device__ void charm::send_dummy_msg(int dst_pe) {
  /*
  size_t msg_size = envelope::alloc_size(0);
  envelope* env = create_envelope(msgtype::dummy, msg_size);

  send_msg(env, msg_size, dst_pe);
  */
}

__device__ void charm::send_reg_msg(int chare_id, int chare_idx, int ep_id,
                                    void* buf, size_t payload_size, int dst_pe) {
  /*
  size_t msg_size = envelope::alloc_size(sizeof(regular_msg) + payload_size);
  envelope* env = create_envelope(msgtype::regular, msg_size);

  regular_msg* msg = new ((char*)env + sizeof(envelope)) regular_msg(chare_id, chare_idx, ep_id);

  // Fill in payload
  if (payload_size > 0) {
    memcpy((char*)msg + sizeof(regular_msg), buf, payload_size);
  }

  send_msg(env, msg_size, dst_pe);
  */
}

__device__ void charm::send_begin_term_msg(int dst_pe) {
  /*
  size_t msg_size = envelope::alloc_size(0);
  envelope* env = create_envelope(msgtype::begin_terminate, msg_size);

  send_msg(env, msg_size, dst_pe);
  */
}

__device__ void charm::send_do_term_msg(int dst_pe) {
  /*
  size_t msg_size = envelope::alloc_size(0);
  envelope* env = create_envelope(msgtype::do_terminate, msg_size);

  send_msg(env, msg_size, dst_pe);
  */
}

__device__ __forceinline__ ssize_t next_msg(void* addr, bool& begin_term_flag,
                                            bool& do_term_flag) {
  /*
  static int dummy_cnt = 0;
  static clock_value_t start;
  static clock_value_t end;
  envelope* env = (envelope*)addr;
#ifdef DEBUG
  printf("PE %d received msg type %d size %llu from PE %d\n",
         nvshmem_my_pe(), env->type, env->size, env->src_pe);
#endif

  if (env->type == msgtype::dummy) {
    // Dummy message
    if (dummy_cnt == 0) {
      start = clock64();
    } else if (dummy_cnt == DUMMY_ITERS-1) {
      end = clock64();
      printf("Receive avg clocks: %lld\n", (end - start) / DUMMY_ITERS);
    }
    dummy_cnt++;
  } else if (env->type == msgtype::create) {
    // Creation message
    create_msg* msg = (create_msg*)((char*)env + sizeof(envelope));
#ifdef DEBUG
    printf("PE %d creation msg chare ID %d, n_local %d, n_total %d, start idx %d, end idx %d\n",
           nvshmem_my_pe(), msg->chare_id, msg->n_local, msg->n_total, msg->start_idx, msg->end_idx);
#endif
    chare_proxy_base*& chare_proxy = chare_proxies[msg->chare_id];
    chare_proxy->alloc(msg->n_local, msg->n_total, msg->start_idx, msg->end_idx);
    char* tmp = (char*)msg + sizeof(create_msg);
    chare_proxy->store_loc_map(tmp);
    tmp += sizeof(int) * msg->n_total;
    for (int i = 0; i < msg->n_local; i++) {
      chare_proxy->unpack(tmp, i);
    }
  } else if (env->type == msgtype::regular) {
    // Regular message
    regular_msg* msg = (regular_msg*)((char*)env + sizeof(envelope));
#ifdef DEBUG
    printf("PE %d regular msg chare ID %d chare idx %d EP ID %d\n", nvshmem_my_pe(), msg->chare_id, msg->chare_idx, msg->ep_id);
#endif
    chare_proxy_base*& chare_proxy = chare_proxies[msg->chare_id];
    void* payload = (char*)msg + sizeof(regular_msg);
    // TODO: Copy payload?
    chare_proxy->call(msg->chare_idx, msg->ep_id, payload);
  } else if (env->type == msgtype::begin_terminate) {
    // Should only be received by PE 0
    assert(my_pe() == 0);
    // Begin termination message
#ifdef DEBUG
    printf("PE %d begin terminate msg\n", nvshmem_my_pe());
#endif
    if (!begin_term_flag) {
      for (int i = 0; i < n_pes(); i++) {
        send_do_term_msg(i);
      }
      begin_term_flag = true;
    }
  } else if (env->type == msgtype::do_terminate) {
    // Do termination message
#ifdef DEBUG
    printf("PE %d do terminate msg\n", nvshmem_my_pe());
#endif
    do_term_flag = true;
  }

  return env->size;
  */
}

__device__ __forceinline__ void recv_msg(int my_pe, int n_pes, bool& begin_term_flag, bool &do_term_flag) {
  /*
  // Retrieve symmetric addresses for metadata
  MsgQueueMeta* send_meta = send_meta_shell[my_pe].meta;
  for (int src_pe = 0; src_pe < n_pes; src_pe++) {
    MsgQueueMeta* recv_meta = recv_meta_shell[src_pe].meta;
    offset_t write = nvshmem_longlong_atomic_fetch(&recv_meta->write, my_pe);
#if DEBUG
    printf("PE %d checking msg queue from PE %d, read %lld, write %lld\n", my_pe, src_pe, recv_meta->read, write);
#endif
    if (recv_meta->read < write) {
      // There are messages to process
      MsgQueueShell& msgq_shell = msg_queue_shell[src_pe];
      ssize_t msg_size = next_msg(msgq_shell.addr(recv_meta->read), begin_term_flag, do_term_flag);
      recv_meta->read += msg_size;
    }
  }
  */
}

__global__ void charm::scheduler(int argc, char** argv, size_t* argvs) {
  /*
  if (!blockIdx.x && !threadIdx.x) {
    bool begin_term_flag = false;
    bool do_term_flag = false;
    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();

    // Register user chares and entry methods on all PEs
    chare_proxy_cnt = 0;
    register_chares();

    // Initialize message queue
    //mpsc_ringbuf_init(rbuf, rbuf_size);
    for (int i = 0; i < n_pes; i++) {
      recv_meta_shell[i].init();
      send_meta_shell[i].init();
    }
    spsc_ringbuf_init(mbuf, mbuf_size);

    nvshmem_barrier_all();

    if (my_pe == 0) {
      // Execute user's main function
      main(argc, argv, argvs);
    }

    nvshmem_barrier_all(); // FIXME: No need?

    // Receive messages and terminate
    do {
      recv_msg(my_pe, n_pes, begin_term_flag, do_term_flag);
    } while (!do_term_flag);

#if DEBUG
    printf("PE %d terminating...\n", my_pe);
#endif
  }
  */
}
