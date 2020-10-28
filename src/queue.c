        #include "queue.h"

        /**
           * @file queue.c
           * @author Anna Lackinger 11776842
           * @date 14.10.2020
           * @brief  queue methods
           * @details this file consists of the methods for the queue
        */


        /**
            * @brief add new node
            * @details insert a new node at the beginning of the list
            * @source original structure taken from https://www.geeksforgeeks.org/doubly-linked-list/
            * @param queue queue to which the new node is added
            * @param task points to a task stored in the node
            * @param start points to the start address of the task
            * @param input_size parameter for executing tasks
        */
        void push(struct Queue* queue, void (*task)(int, Parameters*, int), int start, int input_size){
            struct Node* new_node = (struct Node*)malloc(sizeof(struct Node));
            if( new_node == NULL ) {
                printf("malloc failed - Not enough memory");
                exit(EXIT_FAILURE);
            }
            new_node->input_size=input_size;
            new_node->start=start;
            new_node->task=task;
            new_node->prev = NULL;
            if(queue->head==NULL){
                new_node->next = NULL;
                queue->tail = new_node;
            }else{
                (queue->head)->prev = new_node;
                new_node->next = queue->head;
            }
            queue->head = new_node;
            queue->list_size = queue->list_size+1;
        }

        /**
            * @brief add new node in critical section
            * @details insert a new node on the front of the list, lock head, so that no other thread can access it while inserting new tasks
            * @source original structure taken from https://www.geeksforgeeks.org/doubly-linked-list/
            * @param queue queue to which the new node is added
            * @param task points to a task stored in the node
            * @param start points to the start address of the task
            * @param input_size parameter for executing tasks
        */
        void pushWithLock(struct Queue* queue, void (*task)(int, Parameters*, int), int start, int input_size){
            struct Node* new_node = (struct Node*)malloc(sizeof(struct Node));
            if( new_node == NULL ) {
                printf("malloc failed - Not enough memory");
                exit(EXIT_FAILURE);
            }
            new_node->input_size=input_size;
            new_node->start=start;
            new_node->task=task;
            new_node->lock=1;
            new_node->prev = NULL;
            if(queue->head==NULL){
                new_node->next = NULL;
                queue->tail = new_node;
            }else{
                (queue->head)->prev = new_node;
                new_node->next = queue->head;
            }
            queue->head = new_node;
            if(queue->head->next!=NULL){ //unset lock of last head, so that it can be stolen
                #pragma omp atomic write
                queue->head->next->lock=0;
            }
            #pragma omp atomic update
            queue->list_size = queue->list_size+1;
        }


        /**
            * @brief add new node in critical section
            * @details insert a new node on the front of the list, lock head, so that no other thread can access it while inserting new tasks
            * @source original structure taken from https://www.geeksforgeeks.org/doubly-linked-list/
            * @param node node before which the new one is inserted
            * @param start points to the start address of the task
            * @param task_complexity is an integer that indicates which method is executed by the threads,
            * 3 means more memory is needed for each task
        */
        void pushBeforeNode(struct Queue* queue, struct Node* node, int start, int task_complexity){
            struct Node* new_node = (struct Node*)malloc(sizeof(struct Node));
            if( new_node == NULL ) {
                printf("malloc failed - Not enough memory");
                exit(EXIT_FAILURE);
            }
            int old_input_size=node->input_size;
            if(node->input_size%2==0){
                node->input_size=old_input_size/2;
            }else{
                node->input_size=old_input_size/2+1; //00 111, 00 11 (immer nach n/2)
            }
            new_node->input_size=old_input_size/2;
            new_node-> start = start;
            if(task_complexity==3){
                node->start=start+old_input_size/2*old_input_size/2;
            }
            else{
                node->start=start+old_input_size/2;
            }
            new_node->task=node->task;
            new_node->lock=0;
            new_node->prev = node->prev;
            new_node->next=node;
            if(node==queue->head){
                queue->head=new_node;
            }
            else{
                node->prev->next=new_node;
            }
            node->prev=new_node;
            #pragma omp atomic update
                queue->list_size = queue->list_size+1;
        }

        /**
            * @brief remove nodes in critical section
            * @details remove nodes at the end of the list, lock tail, so that no other thread can access it while removing tasks
            * @param queue queue from which the nodes are taken
            * @param parameters parameter for executing tasks
        */
        void removeTailWithLock(struct Queue* queue, struct Parameters* parameters){
            #pragma omp critical//Only one task can steal from this queue
            {
                if(queue->tail->lock==0){
                    queue->tail->lock=omp_get_thread_num()+2;
                }
            }
            if(queue->tail->lock==omp_get_thread_num()+2){
                int list_size;
                #pragma omp atomic read
                    list_size=queue->list_size;
                if(list_size!=0){
                    int tasks_to_steal;
                    if(list_size>8){
                        tasks_to_steal = list_size/2;
                    }else{
                        tasks_to_steal=list_size;
                    }
                    struct Node* old_tail = queue->tail;
                    struct Node* helpNode = queue->tail;
                    int stolen_tasks=0;
                    int locked;
                    for(;stolen_tasks<tasks_to_steal && stolen_tasks<list_size;stolen_tasks++){
                        #pragma omp atomic write
                            locked = helpNode->lock;
                        if(locked==1){
                            helpNode=helpNode->next;
                            break;
                        }
                        helpNode=helpNode->prev;
                    }
                    #pragma omp atomic
                    queue->list_size = (queue->list_size)-stolen_tasks;
                    if(queue->list_size!=0){
                        helpNode->next->prev=NULL; //separate both
                        helpNode->next=NULL;
                        queue->tail=helpNode;
                    }
                    #pragma omp atomic write
                        queue->tail->lock=0;
                    while(old_tail!=NULL){
                        (* old_tail->task)(old_tail->input_size, parameters, old_tail->start); //execute task
                        old_tail=old_tail->prev;
                    }
                }

            }

        }

        /**
            * @brief add queue
            * @details insert a local queue into the global queue
            * @param global_queue queue to which the new queue is added
            * @param local_queue local queue of one thread
        */
        void pushQueue(struct Global_Queue* global_queue, struct Queue* local_queue){
            local_queue->prev_queue = NULL;
            if(global_queue->head_queue==NULL){
                local_queue->next_queue = NULL;
                global_queue->tail_queue = local_queue;
            }else{
                (global_queue->head_queue)->prev_queue = local_queue;
                local_queue->next_queue = global_queue->head_queue;
            }
            global_queue->head_queue = local_queue;
            global_queue->list_size = global_queue->list_size+1;
        }

        /**
            * @brief initialize queue
            * @param queue queue to initialize
        */
        void init_queue(struct Queue* queue){
            queue->head = NULL;
            queue->tail = NULL;
            queue->next_queue = NULL;
            queue->prev_queue = NULL;
            queue->list_size=0;
        }

        /**
            * @brief initialize global queue
            * @param queue global queue to initialize
        */
        void init_global_queue(struct Global_Queue* queue){
            queue->head_queue = NULL;
            queue->list_size=0;
        }
