    #include <omp.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <getopt.h>
    #include <stdint.h>
    #include <string.h>

    /**
       * @file queue.h
       * @author Anna Lackinger 11776842
       * @date 14.10.2020
       * @brief  header file for queue.c
       * @details this file consists of the structure and methods for the queue
    */

    struct Parameters;
    typedef struct Parameters  Parameters;

    /**
        * @brief elements of the queue
        * @details double linked list, queue points to head and tail
        * @source original structure taken from https://www.geeksforgeeks.org/doubly-linked-list/
        * @element lock, lock head when adding tasks and tail when stealing
                         - 0=unlocked, 1=head, lock>1=rank of thread with id lock-2 currently owns the lock
        * @element next points to next node in DLL
        * @element prev points to prev node in DLL
        * @element task points the task with has to executed by one thread
    */
    struct Node {
        int start;
        int lock;
        struct Node* next;
        struct Node* prev;
        void (*task)(int, Parameters*, int);
    };

     /**
        * @brief parameters for the tasks
        * @details use a struct for the parameters to allocate enough memory before executing the tasks
        * @element A pointer to vector or matrix
        * @element B pointer to vector or matrix
        * @element C pointer to vector or matrix - result of A (operation) B
        * @element start pointer to the beginning of the address, for a thread
        * @element end pointer to the end of the address, for a thread
    */
    struct Parameters {
        double *A;
        double *B;
        double *C;
        int start;
        int end;
    };

    /**
        * @brief queue which is used as a local queue
        * @details double linked list, queue points to head and tail of its nodes
        * @element list_size size of its elements/nodes
        * @element head points to the first node
        * @element tail points to the last node
        * @element next_queue only used for thread accessing other local queues
        * @element prev_queue only used for thread accessing other local queues
    */
    struct Queue {
        int list_size;
        struct Node* head;
        struct Node* tail;
        struct Queue* next_queue;
        struct Queue* prev_queue;
    };

    /**
        * @brief queue which is used as a global queue
        * @details queue points to head and tail of its local queues
        * @element list_size size of its elements/queues
        * @element head_queue points to the first queue
        * @element tail_queue points to the last queue

    */
    struct Global_Queue {
        int list_size;
        struct Queue* head_queue;
        struct Queue* tail_queue;
    };

    //Description can be found in queue.c
    void push(struct Queue* queue, void (*task)(int, Parameters*, int), int start);
    void pushWithLock(struct Queue* queue, void (*task)(int, Parameters*, int), int start);
    void removeTailWithLock(struct Queue* queue, int input_size, struct Parameters* parameters);
    void pushQueue(struct Global_Queue* global_queue, struct Queue* local_queue);
    void init_queue(struct Queue* queue);
    void init_global_queue(struct Global_Queue* queue);
