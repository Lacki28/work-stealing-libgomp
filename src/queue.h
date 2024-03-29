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
        * @element start points to the starting address for executing tasks
        * @element input_size inputparameter for task
        * @element lock, lock head when adding tasks and tail when stealing
                         -> 0=unlocked, 1=head, lock>1= thread with id = lock-2 currently owns the lock
        * @element next points to next node in DLL
        * @element prev points to prev node in DLL
        * @element task points the task which has to executed by one thread
    */
    struct Node {
        int start;
        int input_size;
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
    */
    struct Parameters {
        double *A;
        double *B;
        double *C;
        int start;
    };

    /**
        * @brief queue which is used as a local queue
        * @details double linked list, queue points to head and tail of its nodes
        * @element list_size size of its elements/nodes
        * @element head points to the first node
        * @element tail points to the last node
      */
    struct Queue {
        int list_size;
        struct Node* head;
        struct Node* tail;
    };


    //Description can be found in queue.c
    void push(struct Queue* queue, void (*task)(int, Parameters*, int), int start, int input_size);
    void pushWithLock(struct Queue* queue, void (*task)(int, Parameters*, int), int start, int input_size);
    void pushBeforeNode(struct Queue* queue, struct Node* node, int start, int task_complexity);
    void removeTasksWithLock(struct Queue* queue, struct Parameters* parameters, int steal_size, int head);
    void init_queue(struct Queue* queue);
