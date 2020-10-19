    #include "queue.h"
    //#include "test.h" //Only used once, focus more on performance, therefore not necessary

    /**
    * @file libgomp-test.c
    * @author Anna Lackinger 11776842
    * @date 14.10.2020
    * @brief  multiplies two hexadecimal numbers
    * @details the program creates four child processes, which then recursively do the calculation of the two numbers.
    */

    /**
        * @brief test if threads execute the tasks correctly
    */
    int test_tasks;

    /**
        * @brief initialize parameters
        * @param params parameters to initialize
    */
    void init_parameters(struct Parameters* params){
        params->A = NULL;
        params->B = NULL;
        params->C = NULL;
        params->start = 0;
        params->end=0;
    }

    /**
        * @brief calculate the start and end address for each thread
        * @param iteration_end amount of tasks one thread has to execute
        * @param rank id of the thread
        * @param tasks amount of tasks that have to be executed
        * @param p number of threads
        * @param n input size for the tasks
    */
    void calculateStartAndEndForParams(struct Parameters* parameters, int rank, int tasks, int p, int n){
        if((tasks%p)>rank){
            parameters->start=rank*n+rank*tasks*n/p;
            parameters->end=parameters->start+tasks*n/p;
        }
        else{
            parameters->start=rank*tasks*n/p+n%p;
            parameters->end=parameters->start+tasks*n/p-1;
        }
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

    //Original function is taken from Parallel Computing (184.710)
    void print_vector(double* m, int n) {
        for(int i=0; i<n; i++) {
          printf("%8.4f ", m[i]);
        }
        printf("%d \n", omp_get_thread_num());
        printf("\n");
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
     */
    void vectorVectorSum (int n, struct Parameters* parameters, int start){
        #pragma omp atomic update
            test_tasks++;
        for (int i = start; i < start+n; ++i){
            parameters->C[i] = parameters->A[i] + parameters->B[i];
        }
    }

    /**
        * @brief multiplication of a matrix with a vector
        * @source original function is taken from https://www.appentra.com/parallel-computation-of-matrix-vector-product/
        * @param n size of the vector and the square matrix
        * @param parameters struct with all details needed for execution of the task
    */
    void matrixVectorProduct (int n, struct Parameters* parameters,  int start){
        #pragma omp atomic update
            test_tasks++;
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
    */
    void matrixMatrixProduct(int n, struct Parameters* parameters,  int start){
        #pragma omp atomic update
            test_tasks++;
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

    //TODO: Doc
    void pattern1WithoutWorkStealing(struct Queue* queue, void (*task_func_ptr)(int, Parameters*, int), int tasks,
                                     int input_size, int p, int e, struct Parameters* parameters){
        struct Node* head_next;
        #pragma omp parallel num_threads(p)
        {
             #pragma omp master //first add tasks into (global) queue
            {
                for(int i=0; i<tasks; i++){
                    if(task_func_ptr==&matrixMatrixProduct){
                        push(queue, task_func_ptr, parameters->start+i*input_size*input_size);
                    }else{
                        push(queue, task_func_ptr, parameters->start+i*input_size);
                    }
                }
                head_next = queue->head;
            }
            #pragma omp barrier //wait for master to finish before executing the tasks
            #pragma omp single //Test if all tasks have been added to queue - implicit barrier, so no task can be executed before this test
            {
                if(queue->list_size!=tasks){
                    printf("Not all tasks have been added correctly in pattern1: %d should be %d", queue->list_size, tasks);
                    exit(EXIT_FAILURE);
                }
            }
            #pragma omp for
            for (int i=0;i<tasks;i++){
                struct Node* currentNode;
                #pragma omp critical
                {
                    currentNode = head_next;
                    head_next = head_next->next;
                }
                #pragma omp atomic
                    queue->list_size=queue->list_size-1;
                (* currentNode->task)(input_size, parameters, currentNode->start); //execute task
                free(currentNode);
            }
        }
    }

    /**
        * @brief execute pattern1: global queue
        * @details execute pattern1 with or without work stealing
        * @param queue global queue used by all threads
        * @param task_func_ptr pointer to the task that will be stored in the nodes
        * @param tasks amount of tasks that have to be executed
        * @param input_size input for each task
        * @param p amount of threads that execute the program
        * @param e specify whether work stealing is used
        * @param parameters input for the tasks
    */
    void pattern1(struct Queue* queue, void (*task_func_ptr)(int, Parameters*, int), int tasks, int input_size, int p, int e, struct Parameters* parameters){
        if(e==1){ //No work stealing
            pattern1WithoutWorkStealing(queue, task_func_ptr, tasks, input_size, p, e, parameters);
        }else if (e==2){
            //work stealing
            pattern1WithoutWorkStealing(queue, task_func_ptr, tasks, input_size, p, e, parameters);
        }else{
            exit(EXIT_FAILURE);
        }

    }

    /**
        * @brief execute tasks
        * @details execute tasks and free the nodes afterwards - not thread safe
        * @param queue where the tasks are saved
        * @param iteration_end amount of tasks to be executed
        * @param input_size parameter for tasks
        * @param parameters input for the tasks
    */
    void executeTasks(struct Queue* queue, int iteration_end, int input_size, struct Parameters* parameters){
        struct Node* head_next = queue->head;
        for (int i=0;i<iteration_end;i++){
            struct Node* currentNode;
            currentNode = head_next;
            head_next = head_next->next;
            (* currentNode->task)(input_size, parameters, currentNode->start); //execute task
            queue->list_size=queue->list_size-1;
            free(currentNode);
        }
    }

    //TODO: Doc
    void pattern2WithoutWorkStealing(struct Global_Queue* global_queue, void (*task_func_ptr)(int, Parameters*, int),
                                    int tasks, int input_size, int p, int e, struct Parameters* parameters){
        #pragma omp parallel num_threads(p)
        {
            int rank = omp_get_thread_num();
            struct Queue* local_queue = (struct Queue*)malloc(sizeof(struct Queue));
            testQueueMemory(local_queue);
            init_queue(local_queue);
            #pragma omp for ordered  //add local queues into global one
                for (int i=0; i<p; i++) {
                    #pragma omp ordered
                    {  //printf("Rank: %d\n", rank);
                       pushQueue(global_queue, local_queue);
                    }
                 }
            if(global_queue->list_size!=p){
                printf("Not all tasks have been added correctly in global_queue pattern2: %d should be %d\n", global_queue->list_size, p);
                exit(EXIT_FAILURE);
            }
            int iteration_end = (tasks%p)>rank ? (tasks/p)+1 : tasks/p;
            calculateStartAndEndForParams(parameters, rank, tasks, p, input_size);
            for(int i=0; i<iteration_end; i++){ // add tasks into local queues
                push(local_queue, task_func_ptr, parameters->start+i*input_size);
            }
            if(local_queue->list_size!=iteration_end){
                printf("Not all tasks have been added correctly in local_queue pattern2: %d should be %d \n", local_queue->list_size, iteration_end);
                exit(EXIT_FAILURE);
            }
            executeTasks(local_queue, iteration_end, input_size, parameters);
            free(local_queue);
        }
    }

    //TODO: Doc, teilen und übersichtlicher machen + ansatz 2 queues eine von der ich stehlen kann und eine normale? - Ende speichern, dass in die globale Queue - critical
    void pattern2WithWorkStealing(struct Global_Queue* global_queue, void (*task_func_ptr)(int, Parameters*, int),
                                    int tasks, int input_size, int p, int e, struct Parameters* parameters){
       #pragma omp parallel num_threads(p)
       {
           int rank=omp_get_thread_num();
           struct Queue* local_queue = (struct Queue*)malloc(sizeof(struct Queue));
           testQueueMemory(local_queue);
           init_queue(local_queue);
           #pragma omp for ordered  //add local queues into global one
               for (int i=0; i<p; i++) {
                   #pragma omp ordered
                   {
                      pushQueue(global_queue, local_queue);
                   }
               }
           if(global_queue->list_size!=p){
               printf("Not all tasks have been added correctly in global_queue pattern2: %d should be %d\n", global_queue->list_size, p);
               exit(EXIT_FAILURE);
           }
           int iteration_end = (tasks%p)>rank ? (tasks/p)+1 : tasks/p;
           calculateStartAndEndForParams(parameters, rank, tasks, p, input_size);
           for(int i=0; i<iteration_end; i++){ // add tasks into local queues
               pushWithLock(local_queue, task_func_ptr, parameters->start+i*input_size);
           }
           if(iteration_end!=0){
               #pragma omp atomic write
                   local_queue->head->lock=0;
           }
           if(local_queue->list_size !=iteration_end){
               printf("Not all tasks have been added correctly in local_queue pattern2: %d should be %d \n", local_queue->list_size, iteration_end);
               exit(EXIT_FAILURE);
           }
           //start executing and stealing
           while(local_queue->list_size>0){
                removeTailWithLock(local_queue, input_size, parameters);
           }
           //STEAL OTHER TASKS - from next
           if(local_queue==global_queue->head_queue){
                while(global_queue->tail_queue->list_size>1){
                   printf("STEALING prev: %d\n", omp_get_thread_num());
                    removeTailWithLock(global_queue->tail_queue, input_size, parameters);
                }
           }else{
               while(local_queue->prev_queue->list_size>1){
                    printf("STEALING prev: %d\n", omp_get_thread_num());
                    removeTailWithLock(local_queue->prev_queue, input_size, parameters);
               }
           } //steal from prev
           if(local_queue==global_queue->tail_queue){
               while(global_queue->head_queue->list_size>1){
                   printf("STEALING next: %d\n", omp_get_thread_num());
                   removeTailWithLock(global_queue->head_queue, input_size, parameters);
               }
           }else{
              while(local_queue->next_queue->list_size>1){
                    printf("STEALING next: %d\n", omp_get_thread_num());
                    removeTailWithLock(local_queue->next_queue, input_size, parameters);
              }
           }
           free(local_queue);
        }
    }

    /**
        * @brief execute pattern2: local queues
        * @details execute pattern2 with or without work stealing
        * @param queue global queue used by all threads
        * @param task_func_ptr pointer to the task that will be stored in the nodes
        * @param tasks amount of tasks that have to be executed
        * @param input_size input for each task
        * @param p amount of threads that execute the program
        * @param e specify whether work stealing is used
        * @param parameters input for the tasks
    */
    void pattern2(struct Global_Queue* global_queue, void (*task_func_ptr)(int, Parameters*, int), int tasks, int input_size, int p, int e, struct Parameters* parameters){
            if(e==1){ //No work stealing
                pattern2WithoutWorkStealing(global_queue, task_func_ptr, tasks, input_size, p, e, parameters);
            }else if (e==2){ //Random selection - select one - threashhold...
                pattern2WithWorkStealing(global_queue, task_func_ptr, tasks, input_size, p, e, parameters);
            }else{
                exit(EXIT_FAILURE);
            }
    }

    //TODO: Doc + implement dynamic creation of tasks as discussed in last meeting
    void pattern3(struct Queue* queue, void (*task_func_ptr)(int, Parameters*, int), int tasks, int input_size, int p, int e, struct Parameters* parameters){
        /*if (tasks < 1){
            struct Node* head_next= queue->head;
            #pragma omp for
            for (int i=0;i<queue->list_size;i++){
                struct Node* currentNode;
                #pragma omp critical
                {
                    currentNode = head_next;
                    head_next = head_next->next;
                    queue->list_size=queue->list_size-1;
                }
                (* currentNode->task)(input_size, parameters); //execute task
            }
            return;
        }
        else{
            #pragma omp parallel num_threads(p)
            {
                if(tasks>0){ //add tasks
                    #pragma omp single nowait
                    {
                        if(tasks>0){
                            tasks--;
                            push(queue, task_func_ptr, i*input_size);
                            pattern3(queue, task_func_ptr, tasks, input_size, p, e, parameters);
                        }
                    }
                }
            }
        }*/
    }

     /**
        * @brief execute the main program
        * @details execute the main program as often as rep specifies and calculate average the execution time
        * @param task_func_ptr pointer to the task that will be stored in the nodes
        * @param create specify which pattern will be used
        * @param m amount of tasks that have to be executed
        * @param n input for each task
        * @param execute specify whether work stealing is used
        * @param p amount of threads that execute the program
        * @param rep how often the program is executed
        * @param parameters input for the tasks
    */
    void executeProgram(void (*task_func_ptr)(int, Parameters*, int), int create, int m, int n, int execute, int p, int rep, struct Parameters* parameters){
        double mean=0;
        for(int i=0; i<rep; i++){
            test_tasks=0;
            double start_time = omp_get_wtime();
            if(create==1){
                struct Queue* queue = (struct Queue*)malloc(sizeof(struct Queue));
                testQueueMemory(queue);
                init_queue(queue);
                pattern1 (queue, task_func_ptr, m, n, p, execute, parameters);
                free(queue);
            }else if(create==2){ //TODO: threshold
                struct Global_Queue* queue = (struct Global_Queue*)malloc(sizeof(struct Global_Queue));
                if( queue == NULL ) {
                    printf("malloc failed - Not enough memory");
                    exit(EXIT_FAILURE);
                }
                init_global_queue(queue);
                pattern2 (queue, task_func_ptr, m, n, p, execute, parameters);
                free(queue);
            }else if(create==3){
               // pattern3 (queue, task_func_ptr, m, n, p, execute, parameters);
            }else{
                exit(EXIT_FAILURE);
            }
            double time = omp_get_wtime() - start_time;
            mean+=time;
            if(test_tasks!=m){
                printf("The number of executed tasks (%d) does not match the required number (%d)\n", test_tasks, m);
                exit(EXIT_FAILURE);
            }
        }
        printf("Done in an average time of: %f\n", (mean/rep));
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
    */
    void createParametersForProgramAndExecute(int create, int m, int n, int execute, int p, int rep, int type) {
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
        executeProgram(task_func_ptr, create, m, n, execute, p, rep, parameters);
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
        if(argc!=15){
            printf("ERROR: libgomp-test -p x --create x --type x -m x -n x --execute x --rep x");
            exit(EXIT_FAILURE);
        }
        for(int i=1; i<argc; i+=2){
            char* arg = argv[i];
            if(atoi(argv[i+1])==0){
                printf("ERROR: libgomp-test -p x --create x --type x -m x -n x --execute x --rep x, parameters must be > 1");
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
            }else{
                printf("\n|%s|\n", arg);
                printf("ERROR: libgomp-test -p x --create x --type x -m x -n x --execute x --rep x");
                exit(EXIT_FAILURE);
            }
        }
        createParametersForProgramAndExecute(create, m, n, execute, p, rep, type);
    }
