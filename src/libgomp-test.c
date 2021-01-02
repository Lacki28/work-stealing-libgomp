        #include "queue.h"

        /**
        * @file libgomp-test.c
        * @author Anna Lackinger 11776842
        * @date 14.10.2020
        * @brief the program creates and executes tasks in parallel
        * @details the program creates and executes tasks in parallel using
        * different creation patterns and stealing methods
        */

        /**
            * @brief test if threads execute the tasks correctly
        */
        int executed_tasks;
        int total_number_of_tasks;
        int all_queues_empty;

        /**
            * @brief parameters used in all patterns
            * @element task_func_ptr pointer to the task that will be stored in the nodes
            * @element tasks amount of tasks that have to be executed
            * @element input_size input for each task
            * @element p amount of threads that execute the program
        */
        struct Function_Parameters{
            void (*task_func_ptr)(int, Parameters*, int);
            int tasks;
            int input_size;
            int p;
        };

        /**
            * @brief initialize parameters
            * @param params parameters to initialize
        */
        void init_parameters(struct Parameters* params){
            params->A = NULL;
            params->B = NULL;
            params->C = NULL;
            params->start = 0;
        }

        /**
            * @brief calculate the start and end address for each thread
            * @param rank id of the thread
            * @param tasks amount of tasks that have to be executed
            * @param p number of threads
            * @param n input size for the tasks
        */
        int calculateStart(int rank, int tasks, int p, int input_size){
            int start=0;
           if(tasks<=p){
               start=rank*input_size;
           }
           else{
               if((tasks%p)>rank){
                    start=rank*input_size*(tasks/p)+(rank*input_size);
               }
               else{
                   start=rank*input_size*(tasks/p)+tasks%p*input_size;
               }
           }
           return start;
        }

        /**
            * @brief fill vector with double 1.0
            * @source original structure taken from Parallel Computing (184.710)
            * @param m vector to fill
            * @param n size of vector
        */
        void fill_vector(double* m, int n) {
            for(int i=0; i<n; i++) {
                m[i] = 1.0;
            }
        }

        /**
            * @brief free vector
            * @source original structure taken from Parallel Computing (184.710)
            * @param m vector to free
        */
        int free_vector(double* m) {
            if( m != NULL ) {
              free(m);
            }
            return 0;
        }

         /**
            * @brief test if malloc() did not fail
            * @param A pointer to vector that has been created
            * @param B pointer to vector that has been created
            * @param C pointer to vector that has been created
            * @param parameters struct that has been created
        */
        void testTaskParamMemory(double *A, double *B, double *C, struct Parameters* parameters){
            if( A == NULL || B == NULL || C == NULL || parameters == NULL) {
                printf("malloc failed - Not enough memory for tasks");
                exit(EXIT_FAILURE);
            }
        }

        //TASKS

        /**
            * @brief addition of two vectors
            * @source original function is taken from https://riptutorial.com/openmp/example/23425/addition-of-two-vectors-using-openmp-parallel-for-construct
            * @param n size of the vectors
            * @param parameters struct with all details needed for execution of the task
            * @param start pointer to the beginning of the address
         */
        void vectorVectorSum (int n, struct Parameters* parameters, int start){
            #pragma omp atomic update
                executed_tasks++;
            for (int i = start; i < start+n; ++i){
                parameters->C[i] = parameters->A[i] + parameters->B[i];
            }
        }

        /**
            * @brief multiplication of a matrix with a vector
            * @source original function is taken from https://www.appentra.com/parallel-computation-of-matrix-vector-product/
            * @param n size of the vector and the square matrix
            * @param parameters struct with all details needed for execution of the task
            * @param start pointer to the beginning of the address
        */
        void matrixVectorProduct (int n, struct Parameters* parameters,  int start){
            #pragma omp atomic update
                executed_tasks++;
            for(int i=0;i<n;i++) {
              parameters->C[start+i]=0;
              for(int j=0;j<n;j++) {
                  parameters->C[start+i] += parameters->B[start+i*n+j]*parameters->A[start+j];
              }
            }
        }

        /**
            * @brief multiplication of two matrices
            * @source original function is taken from https://www.appentra.com/parallel-matrix-matrix-multiplication/
            * @param n size of the square matrices
            * @param parameters struct with all details needed for execution of the task
            * @param start pointer to the beginning of the address
        */
        void matrixMatrixProduct(int n, struct Parameters* parameters,  int start){
            #pragma omp atomic update
                executed_tasks++;
            for (int i=0; i<n; i=i+1){
                for (int j=0; j<n; j=j+1){
                        parameters->C[start+i*n+j]=0.0;
                     for (int k=0; k<n; k=k+1){
                        parameters->C[start+i*n+j]+=(parameters->A[start+i*n+k])*(parameters->B[start+k*n+j]);
                     }
                }
            }
        }

        /**
            * @brief check if the allocation of the queue had no errors
            * @param queue queue which has been created
        */
        void testQueueMemory(struct Queue* queue){
            if( queue == NULL ) {
                    printf("malloc failed - Not enough memory");
                    exit(EXIT_FAILURE);
            }
        }

        /**
            * @brief execute tasks that are stored in global queue
            * @details parallel execution of tasks stored in a global queue, without work stealing
            * @param queue global queue used by all threads
            * @param fp parameters all patterns use
            * @param parameters input for the tasks
        */
        double pattern1WithoutWorkStealing(struct Queue* queue, struct Function_Parameters* fp, struct Parameters* parameters){
            struct Node* head_next;
            double start_time;
            #pragma omp parallel num_threads(fp->p)
            {
                 #pragma omp master //first add tasks into (global) queue
                {
                    for(int i=0; i<fp->tasks; i++){
                        if(fp->task_func_ptr==&matrixMatrixProduct){
                            push(queue, fp->task_func_ptr, parameters->start+i*fp->input_size*fp->input_size,fp->input_size);
                        }else{
                            push(queue, fp->task_func_ptr, parameters->start+i*fp->input_size, fp->input_size);
                        }
                    }
                    head_next = queue->head;
                }
                #pragma omp barrier //wait for master to finish before executing the tasks
                #pragma omp single //Test if all tasks have been added to queue - implicit barrier, so no task can be executed before this test
                {
                    if(queue->list_size!=fp->tasks){
                        printf("Not all tasks have been added correctly in pattern1: %d should be %d", queue->list_size, fp->tasks);
                        exit(EXIT_FAILURE);
                    }
                }
                #pragma omp barrier
                #pragma omp single
                   start_time = omp_get_wtime();
                #pragma omp for
                for (int i=0;i<fp->tasks;i++){
                    struct Node* currentNode;
                    #pragma omp critical
                    {
                        currentNode = head_next;
                        head_next = head_next->prev;
                    }
                    #pragma omp atomic update
                        queue->list_size=queue->list_size-1;
                    (* currentNode->task)(currentNode->input_size, parameters, currentNode->start); //execute task

                }
                #pragma omp master
                    all_queues_empty=0;
            }
            return omp_get_wtime()-start_time;
        }

        /**
            * @brief execute pattern1: global queue
            * @details execute pattern1 with or without work stealing
            * @param queue global queue used by all threads
            * @param function_Parameters parameters all patterns use
            * @param e specify whether work stealing is used
            * @param parameters input for the tasks
        */
        double pattern1(struct Queue* queue, struct Function_Parameters* function_Parameters, int e, struct Parameters* parameters){
            if(e==1){ //No work stealing
                return pattern1WithoutWorkStealing(queue, function_Parameters, parameters);
            }else if (e==2||e==3||e==4||e==5){
                //work stealing
                return 0;
            }else{
                exit(EXIT_FAILURE);
            }

        }

        /**
            * @brief execute tasks
            * @details execute tasks - not thread safe
            * @param queue where the tasks are saved
            * @param iteration_end amount of tasks to be executed
            * @param parameters input for the tasks
        */
        void executeTasks(struct Queue* queue, int iteration_end, struct Parameters* parameters){
            struct Node* head_next = queue->head;
            while (queue->list_size>0){
                struct Node* currentNode;
                currentNode = head_next;
                head_next = head_next->prev;
                (* currentNode->task)(currentNode->input_size, parameters, currentNode->start); //execute task
                queue->list_size=queue->list_size-1;
            }
        }

         /**
            * @brief execute tasks that are stored in local queues
            * @details Each thread creates its own queue in which the tasks are stored for later execution.
            * Without the need for synchronization, the threads then perform the tasks of their own queues.
            * @param global_array global array used by all threads
            * @param fp parameters all patterns use
            * @param parameters input for the tasks
        */
        double pattern2WithoutWorkStealing(struct Queue* global_array, struct Function_Parameters* fp, struct Parameters* parameters){
            double start_time = omp_get_wtime();
            #pragma omp parallel num_threads(fp->p)
            {
                int rank = omp_get_thread_num();
                struct Queue* local_queue = (struct Queue*)malloc(sizeof(struct Queue));
                testQueueMemory(local_queue);
                init_queue(local_queue);
                #pragma omp for ordered  //add local queues to global array
                    for (int i=0; i<fp->p; i++) {
                        #pragma omp ordered
                        {  //printf("Rank: %d\n", rank);
                           global_array[i] = *local_queue;
                        }
                    }
                int iteration_end = (fp->tasks%fp->p)>rank ? (fp->tasks/fp->p)+1 : fp->tasks/fp->p;
                if(iteration_end!=0){
                    int start;
                    if(fp->task_func_ptr==&matrixMatrixProduct){
                        start= calculateStart(rank, fp->tasks, fp->p, fp->input_size*fp->input_size);
                    }else{
                        start= calculateStart(rank, fp->tasks, fp->p, fp->input_size);
                    }
                    for(int i=0; i<iteration_end; i++){ // add tasks to local queues
                        if(fp->task_func_ptr==&matrixMatrixProduct){
                             push(local_queue, fp->task_func_ptr, start+i*fp->input_size*fp->input_size, fp->input_size);
                        }else{
                             push(local_queue, fp->task_func_ptr, start+i*fp->input_size, fp->input_size);
                        }
                    }
                    if(local_queue->list_size!=iteration_end){
                        printf("Not all tasks have been added correctly in local_queue pattern2: %d should be %d \n",
                        local_queue->list_size, iteration_end);
                        exit(EXIT_FAILURE);
                    }
                }
                #pragma omp barrier
                #pragma omp single
                    start_time = omp_get_wtime();
                executeTasks(local_queue, iteration_end, parameters);
                #pragma omp atomic update
                    all_queues_empty--;
            }
            return omp_get_wtime()-start_time;
        }

        /**
            * @brief steal tasks
            * @details randomly choose one queue to steal from
            * @param global_array stores the local queues of all tasks
            * @param parameters input for the tasks
            * @param p number of threads
            * @param id of the thread
            * @param e specifies the steal strategy
        */
        void stealTasks(struct Queue *global_array, struct Parameters* parameters, int p, int id, int e){
            if(e==2){
                while(all_queues_empty!=0){
                int thread_to_steal_from =  rand() % p;
                    if(global_array[thread_to_steal_from].list_size!=0){
                       removeTasksWithLock(&global_array[thread_to_steal_from], parameters, 1, 0);
                    }
                }
            }
            else{
                while(all_queues_empty!=0){
                    int thread_to_steal_from =  rand() % p;
                    if(global_array[thread_to_steal_from].list_size!=0){
                       removeTasksWithLock(&global_array[thread_to_steal_from], parameters, global_array[thread_to_steal_from].list_size/2, 0);
                    }
                }
            }
        }

         /**
            * @brief steal tasks
            * @details randomly select two queues and steal tasks from the queue with more tasks
            * @param global_array stores the local queues of all tasks
            * @param parameters input for the tasks
            * @param p number of threads
            * @param id of the thread
            * @param e specifies the steal strategy
        */
        void stealTasksFromTwo(struct Queue *global_array, struct Parameters* parameters, int p, int id, int e){
            while(all_queues_empty!=0){
                int steal_size=1;
                int thread_to_steal_from1 =  rand() % p;
                int thread_to_steal_from2 =  rand() % p;
                int max_queue=thread_to_steal_from2;
                if(global_array[thread_to_steal_from1].list_size>=global_array[thread_to_steal_from2].list_size){
                    max_queue=thread_to_steal_from1;
                }
                if(global_array[max_queue].list_size!=0){
                    if(e==5){
                       steal_size=global_array[max_queue].list_size/2;
                    }
                    removeTasksWithLock(&global_array[max_queue], parameters, steal_size, 0);
                }
            }
        }

         /**
            * @brief execute tasks stored in local queues and steal tasks from neighboring threads if possible
            * @details Each thread creates its own queue in which the tasks are stored for later execution.
            * The queues of all threads are stored in a global queue that each thread can access.
            * After one thread has completed its own tasks it then tries to steal tasks from neighboring threads if possible
            * and executes the stoles tasks.
            * @param global_array global array used by all threads
            * @param fp parameters all patterns use
            * @param parameters input for the tasks
            * @param e indicates which work stealing method is used
        */
        double pattern2WithWorkStealing(struct Queue* global_array, struct Function_Parameters* fp, struct Parameters* parameters, int e){
            double start_time = omp_get_wtime();
            #pragma omp parallel num_threads(fp->p)
            {
                int rank=omp_get_thread_num();
                struct Queue* local_queue = (struct Queue*)malloc(sizeof(struct Queue));
                testQueueMemory(local_queue);
                init_queue(local_queue);
                #pragma omp for ordered  //add local queues to global array
                    for (int i=0; i<fp->p; i++) {
                        #pragma omp ordered
                        {
                           global_array[i] = *local_queue;
                        }
                    }
                int iteration_end = (fp->tasks%fp->p)>rank ? (fp->tasks/fp->p)+1 : fp->tasks/fp->p;
                if(iteration_end!=0){
                    int start;
                    if(fp->task_func_ptr==&matrixMatrixProduct){
                        start= calculateStart(rank, fp->tasks, fp->p, fp->input_size*fp->input_size);
                    }else{
                        start= calculateStart(rank, fp->tasks, fp->p, fp->input_size);
                    }
                    for(int i=0; i<iteration_end; i++){ // add tasks to local queues
                        if(fp->task_func_ptr==&matrixMatrixProduct){
                             pushWithLock(local_queue, fp->task_func_ptr, start+i*fp->input_size*fp->input_size, fp->input_size);
                        }else{
                             pushWithLock(local_queue, fp->task_func_ptr, start+i*fp->input_size, fp->input_size);
                        }
                    }
                    if(local_queue->list_size!=iteration_end){
                        printf("Not all tasks have been added correctly in local_queue pattern2: %d should be %d \n",
                        local_queue->list_size, iteration_end);
                        exit(EXIT_FAILURE);
                    }
                }
                #pragma omp barrier
                #pragma omp single
                    start_time = omp_get_wtime();
                while(local_queue->list_size>0){
                    removeTasksWithLock(local_queue, parameters, 1, 1);
                }
                #pragma omp atomic update
                    all_queues_empty--;
                if(fp->p<fp->tasks){
                    if(e==2||e==3)
                        stealTasks(global_array, parameters, fp->p, rank, e);
                    else
                        stealTasksFromTwo(global_array, parameters, fp->p, rank, e);
                }
            }
            return omp_get_wtime()-start_time;
        }

        /**
            * @brief execute pattern2: local queues
            * @details execute pattern2 with or without work stealing
            * @param global_array global array used by all threads
            * @param function_Parameters parameters all patterns use
            * @param e specify whether work stealing is used
            * @param parameters input for the tasks
        */
        double pattern2(struct Queue* global_array, struct Function_Parameters* function_Parameters, int e, struct Parameters* parameters){
            if(e==1){ //No work stealing
                return pattern2WithoutWorkStealing(global_array,function_Parameters, parameters);
            }else if (e==2||e==3||e==4||e==5){
                return pattern2WithWorkStealing(global_array, function_Parameters, parameters, e);
            }else{
                exit(EXIT_FAILURE);
            }
        }


         /**
            * @brief double nodes and halve input size
            * @details each node is replaced by two new nodes having half of the original input size
            * @param local_queue queue in which the nodes are stored
            * @param parameters input for the tasks
            * @param start start address of the first node in the queue
            * @param input_size input for each task
        */
        void doubleTasks(struct Queue* local_queue, struct Parameters* parameters,
        int start, int input_size){
            struct Node* current_Node =local_queue->head;
            while(current_Node!=NULL){
                if(current_Node-> task== &matrixMatrixProduct){
                    pushBeforeNode(local_queue, current_Node, current_Node->start, 3);
                }
                else{
                    pushBeforeNode(local_queue, current_Node, current_Node->start, 1);
                }
                #pragma omp atomic update
                    total_number_of_tasks+=1;
                current_Node=current_Node->prev;
            }
        }

         /**
            * @brief double nodes
            * @details If the input size is below a certain threshold, split each node
            * and make two new nodes that store half of the input size.
            * @param local_queue queue used by one thread
            * @param input_size input for each task
            * @param parameters input for the tasks
            * @param start address stored in the header node to execute the tasks
            * @param task_func_ptr pointer to the task that will be stored in the nodes

        */
        void doubleTasksRecursively(struct Queue* local_queue, int input_size, struct Parameters* parameters, int start,
        void (*task_func_ptr)(int, Parameters*, int)){
            if(input_size>10){
               doubleTasks(local_queue, parameters, start, input_size);
               doubleTasksRecursively(local_queue, input_size/2, parameters, start, task_func_ptr);
            }
        }

         /**
            * @brief execute tasks stored in local queues
            * @details Each thread creates its own queue in which the tasks are stored for later execution.
            * The input size of the tasks are then halved and their nodes doubled until they are below a certain threshold
            * The queues of all threads are stored in a global queue that each thread can access.
            * After one thread has completed building its queue with all nodes having an input size below a certain threshold it then executes
            * its own tasks.
            * @param global_array global array used by all threads
            * @param fp parameters all patterns use
            * @param parameters input for the tasks
        */
        double pattern3WithoutWorkStealing(struct Queue* global_array, struct Function_Parameters* fp, struct Parameters* parameters){
             double start_time = omp_get_wtime();
             #pragma omp parallel num_threads(fp->p)
            {
               int rank=omp_get_thread_num();
               struct Queue* local_queue = (struct Queue*)malloc(sizeof(struct Queue));
               testQueueMemory(local_queue);
               init_queue(local_queue);
               #pragma omp for ordered  //add local queues to global array
                   for (int i=0; i<fp->p; i++) {
                       #pragma omp ordered
                       {
                          global_array[i] = *local_queue;
                       }
                   }
               int iteration_end = (fp->tasks%fp->p)>rank ? (fp->tasks/fp->p)+1 : fp->tasks/fp->p;
               if(iteration_end!=0){
                   int start;
                   if(fp->task_func_ptr==&matrixMatrixProduct){
                       start= calculateStart(rank, fp->tasks, fp->p, fp->input_size*fp->input_size);
                   }else{
                       start= calculateStart(rank, fp->tasks, fp->p, fp->input_size);
                   }
                   for(int i=0; i<iteration_end; i++){ // add tasks into local queues
                       if(fp->task_func_ptr==&matrixMatrixProduct){
                            push(local_queue, fp->task_func_ptr, start+i*fp->input_size*fp->input_size,fp->input_size);
                       }else{
                            push(local_queue, fp->task_func_ptr, start+i*fp->input_size, fp->input_size);
                       }
                   }
                   if(local_queue->list_size!=iteration_end){
                       printf("Not all tasks have been added correctly in local_queue pattern2: %d should be %d \n",
                       local_queue->list_size, iteration_end);
                       exit(EXIT_FAILURE);
                   } //split tasks
                   doubleTasksRecursively(local_queue, fp->input_size, parameters, start, fp->task_func_ptr);
               }
               #pragma omp barrier
               #pragma omp single
                   start_time = omp_get_wtime();
               executeTasks(local_queue, iteration_end, parameters);
               #pragma omp atomic update
                    all_queues_empty--;
            }
            return omp_get_wtime()-start_time;
        }

        /**
            * @brief execute tasks stored in local queues
            * @details Each thread creates its own queue in which the tasks are stored for later execution.
            * The input size of the tasks are then halved and their nodes doubled until they are below a certain threshold
            * The queues of all threads are stored in a global queue that each thread can access.
            * After one thread has completed building its queue with all nodes having an input size below a certain threshold it then executes
            * its own tasks and afterwards tries to steal tasks from neighboring threads if possible
            * and executes the stoles tasks.
            * @param global_array global array used by all threads
            * @param fp parameters all patterns use
            * @param parameters input for the tasks
            * @param e indicates which work stealing method is used
        */
        double pattern3WithWorkStealing(struct Queue* global_array, struct Function_Parameters* fp, struct Parameters* parameters, int e){
            double start_time = omp_get_wtime();
            #pragma omp parallel num_threads(fp->p)
            {
               int rank=omp_get_thread_num();
               struct Queue* local_queue = (struct Queue*)malloc(sizeof(struct Queue));
               testQueueMemory(local_queue);
               init_queue(local_queue);
               #pragma omp for ordered  //add local queues to global array
                    for (int i=0; i<fp->p; i++) {
                        #pragma omp ordered
                        {
                            global_array[i] = *local_queue;
                        }
                    }
               int iteration_end = (fp->tasks%fp->p)>rank ? (fp->tasks/fp->p)+1 : fp->tasks/fp->p;
               if(rank<fp->tasks){
                   int start;
                   if(fp->task_func_ptr==&matrixMatrixProduct){
                       start= calculateStart(rank, fp->tasks, fp->p, fp->input_size*fp->input_size);
                   }else{
                       start= calculateStart(rank, fp->tasks, fp->p, fp->input_size);
                   }
                   pushWithLock(local_queue, fp->task_func_ptr, start, fp->input_size);//lock tail
                   for(int i=1; i<iteration_end; i++){ // add tasks into local queues
                       if(fp->task_func_ptr==&matrixMatrixProduct){
                            push(local_queue, fp->task_func_ptr, start+i*fp->input_size*fp->input_size,fp->input_size);
                       }else{
                            push(local_queue, fp->task_func_ptr, start+i*fp->input_size, fp->input_size);
                       }
                   }
                   //start executing and stealing
                   doubleTasksRecursively(local_queue, fp->input_size, parameters, start, fp->task_func_ptr);
               }
               #pragma omp barrier
               #pragma omp single
                   start_time = omp_get_wtime();
               if(iteration_end!=0){
                   while(local_queue->list_size>0){
                       removeTasksWithLock(local_queue, parameters, 1, 1);
                   }
                   #pragma omp atomic update
                        all_queues_empty--;
                   if(e==2||e==3)
                       stealTasks(global_array, parameters, fp->p, rank, e);
                   else
                       stealTasksFromTwo(global_array, parameters, fp->p, rank, e);
               }
            }
            return omp_get_wtime()-start_time;
        }


        /**
            * @brief execute pattern3: local queues
            * @details execute pattern3 with or without work stealing, create tasks recursively
            * @param queue global queue used by all threads
            * @param function_Parameters parameters all patterns use
            * @param e specify whether work stealing is used
            * @param parameters input for the tasks
        */
        double pattern3(struct Queue* global_array, struct Function_Parameters* function_Parameters,
         int e, struct Parameters* parameters){
            if(e==1){
                return pattern3WithoutWorkStealing(global_array, function_Parameters, parameters);
            }else if (e==2||e==3||e==4||e==5){
                return pattern3WithWorkStealing(global_array, function_Parameters, parameters, e);
            }else{
                exit(EXIT_FAILURE);
            }
        }

         /**
            * @brief execute the main program
            * @details execute the main program as often as rep specifies and calculate average the execution time
            * @param create specify which pattern will be used
            * @param function_Parameters parameters all patterns use
            * @param execute specify whether work stealing is used
            * @param rep how often the program is executed
            * @param parameters input for the tasks
            * @param f name of outputfile
        */

        void executeProgram(int create, struct Function_Parameters* function_Parameters, int execute,
        int rep, struct Parameters* parameters, char * f){
            double execution_mean=0;
            double execution_time=0;
            double min = 0;
            double max = 0;
            //Write into three files
            char* filename = ".txt";
            char* filenameMin = "min.txt";
            char* filenameMax = "max.txt";
            char * file = (char *) malloc(1 + strlen(f)+ strlen(filename) );
            char * filemin = (char *) malloc(1 + strlen(f)+ strlen(filenameMin) );
            char * filemax = (char *) malloc(1 + strlen(f)+ strlen(filenameMax) );
            strcpy(file, f);
            strcat(file, filename);
            strcpy(filemin, f);
            strcat(filemin, filenameMin);
            strcpy(filemax, f);
            strcat(filemax, filenameMax);
            FILE *fp = fopen(file, "a");
            FILE *fp_min = fopen(filemin, "a");
            FILE *fp_max = fopen(filemax, "a");
            for(int i=0; i<rep; i++){
                executed_tasks=0;
                all_queues_empty = function_Parameters->p;
                total_number_of_tasks=function_Parameters->tasks;;
                if(create==1){
                    struct Queue* queue = (struct Queue*)malloc(sizeof(struct Queue));
                    testQueueMemory(queue);
                    init_queue(queue);
                    execution_time=pattern1 (queue, function_Parameters, execute, parameters);
                    free(queue);
                }else if(create==2){
                    struct Queue *global_array = malloc(function_Parameters->p * sizeof(struct Queue));
                    if( global_array == NULL ) {
                        printf("malloc failed - Not enough memory");
                        exit(EXIT_FAILURE);
                    }
                    execution_time=pattern2(global_array, function_Parameters, execute, parameters);
                    free(global_array);
                }else if(create==3){
                    struct Queue *global_array = malloc(function_Parameters->p * sizeof(struct Queue));
                    if( global_array == NULL ) {
                        printf("malloc failed - Not enough memory");
                        exit(EXIT_FAILURE);
                    }
                    execution_time=pattern3 (global_array, function_Parameters, execute, parameters);
                    free(global_array);
                }else{
                    exit(EXIT_FAILURE);
                }
                execution_mean+=execution_time;
                if(execution_time<min || min == 0){
                    min = execution_time;
                }
                if(execution_time>max || max == 0){
                    max = execution_time;
                }
                if(executed_tasks!=total_number_of_tasks){
                    if (!(create == 1 && (execute ==2||execute ==3||execute ==4||execute ==5))){
                        printf("The number of executed tasks (%d) does not match the required number (%d)\n", executed_tasks, total_number_of_tasks);
                        exit(EXIT_FAILURE);
                    }
                }
                if(all_queues_empty!=0){
                    printf("A queue still has unfinished tasks!\n");
                }
            }
            fprintf(fp, "%f, ", (execution_mean/rep));
            fprintf(fp_min, "%f, ", min);
            fprintf(fp_max, "%f, ", max);
        }

        /**
            * @brief create parameters for the program and execute it
            * @param create specify which pattern will be used
            * @param m amount of tasks that have to be executed
            * @param n input for each task
            * @param execute specify whether work stealing is used
            * @param p amount of threads that execute the program
            * @param rep how often the program is executed
            * @param type kind of task
            * @param f name of outputfile
        */
        void createParametersForProgramAndExecute(int create, int m, int n, int execute, int p, int rep, int type, char* f) {
            //Create task pointer here and input Params
            void (*task_func_ptr)(int, Parameters*,  int);
            int A_size=0;
            int B_size=0;
            int C_size=0;
            if(type==1){
                task_func_ptr =  &vectorVectorSum;
                A_size=n*m;
                B_size=n*m;
                C_size=n*m;
            }else if(type==2){
                task_func_ptr = &matrixVectorProduct;
                A_size=n*m;
                B_size=n*n*m;
                C_size=n*m;
            }else if(type==3){
                task_func_ptr = &matrixMatrixProduct;
                A_size=n*n*m;
                B_size=n*n*m;
                C_size=n*n*m;
            }else{
                exit(EXIT_FAILURE);
            }
            struct Function_Parameters* function_Parameters = (struct Function_Parameters*)malloc(sizeof(struct Function_Parameters));
            function_Parameters->task_func_ptr = task_func_ptr;
            function_Parameters->tasks=m;
            function_Parameters->input_size=n;
            function_Parameters->p=p;

            double *A=(double*)malloc(A_size * sizeof(double));
            double *B=(double*)malloc(B_size * sizeof(double));
            double *C=(double*)malloc(C_size * sizeof(double));
            struct Parameters* parameters = (struct Parameters*)malloc(sizeof(struct Parameters));
            testTaskParamMemory(A,B,C, parameters);
            init_parameters(parameters);
            fill_vector(A, A_size);
            fill_vector(B, B_size);
            fill_vector(C, C_size);
            parameters->A=A;
            parameters->B=B;
            parameters->C=C;
            executeProgram(create, function_Parameters, execute, rep, parameters, f);
            free_vector(A);
            free_vector(B);
            free_vector(C);
        }
         /**
            * @brief check parameters and start executing the program
            * @details exit with failure if one parameter is incorrect
            * @param argc number of arguments
            * @param argv value of the arguments
            * @return int exit status
        */
        int main(int argc, char **argv) {
            int p = -1;
            int create = -1;
            int type = -1;
            int m = -1;
            int n = -1;
            int execute = -1;
            int rep=-1;
            char *f;
            if(argc!=17){
                printf("ERROR: libgomp-test -p x -create x -type x -m x -n x -execute x -rep x -f x");
                exit(EXIT_FAILURE);
            }
            for(int i=1; i<argc; i+=2){
                char* arg = argv[i];
                if(atoi(argv[i+1])==0 && i!=15){
                    printf("ERROR: libgomp-test -p x -create x -type x -m x -n x -execute x -rep x -f x, missing parameter after: %s", argv[i]);
                    exit(EXIT_FAILURE);
                }
                if(strcmp(arg, "-p")==0){
                    p = atoi(argv[i+1]);
                }else if(strcmp(arg, "-create")==0){
                    create = atoi(argv[i+1]);
                }else if(strcmp(arg, "-type")==0){
                    type = atoi(argv[i+1]);
                }else if(strcmp(arg, "-m")==0){
                    m = atoi(argv[i+1]);
                }else if(strcmp(arg, "-n")==0){
                    n = atoi(argv[i+1]);
                }else if(strcmp(arg, "-execute")==0){
                    execute = atoi(argv[i+1]);
                }else if(strcmp(arg, "-rep")==0){
                    rep = atoi(argv[i+1]);
                }else if(strcmp(arg, "-f")==0){
                    f = argv[i+1];
                }else{
                    printf("\n|%s|\n", arg);
                    printf("ERROR: libgomp-test -p x -create x -type x -m x -n x -execute x -rep x -f x");
                    exit(EXIT_FAILURE);
                }
            }
            createParametersForProgramAndExecute(create, m, n, execute, p, rep, type, f);
        }
