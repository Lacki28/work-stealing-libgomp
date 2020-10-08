    #include <omp.h>
    #include <stdio.h>
    #include <stdlib.h>    /* for exit */
    #include <getopt.h>
    #include <stdint.h>
    #include <string.h>

    //Original structure taken from https://www.geeksforgeeks.org/doubly-linked-list/
    struct Node {
        int lock; //lock head when adding tasks and queue when stealing
        struct Node* next; // Pointer to next node in DLL
        struct Node* prev; // Pointer to previous node in DLL
        void (*task)(int); // task
    };

    struct Queue {
        int list_size;
        int index;
        struct Node* head; // Pointer to first node in DLL
        struct Node* tail; //  Pointer to last node in DLL
        struct Queue* next_queue; // Only used in global queues
        struct Queue* prev_queue; // Only used in global queues
    };

    struct Global_Queue {
            int list_size;
            struct Queue* head_queue; // Pointer to first Queue in DLL
            struct Queue* tail_queue; // Pointer to first Queue in DLL
    };


    //Original structure taken from https://www.geeksforgeeks.org/doubly-linked-list/
    /* Given a reference (pointer to pointer) to the head of a list
       and an int, inserts a new node on the front of the list. */
    void push(struct Queue* queue, void (*task)(int)){
        struct Node* new_node = (struct Node*)malloc(sizeof(struct Node));
        if( new_node == NULL ) {
            printf("malloc failed - Not enough memory");
            exit(EXIT_FAILURE);
        }
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
        //printf("%d\n", queue->list_size);
    }


    void pushWithLock(struct Queue* queue, void (*task)(int)){
        struct Node* new_node = (struct Node*)malloc(sizeof(struct Node));
        if( new_node == NULL ) {
            printf("malloc failed - Not enough memory");
            exit(EXIT_FAILURE);
        }
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
        #pragma omp atomic
        queue->list_size = queue->list_size+1;
    }

    void removeTailWithLock(struct Queue* queue, int input_size){
        struct Node* old_tail;
        if(queue->list_size!=0)
        #pragma omp critical (critical_section)//Only one task can steal from this queue
        {
            int list_size;
            #pragma omp atomic read
                list_size=queue->list_size;
            if(list_size!=0){
                int tasks_to_steal;
                if(list_size>5){
                    tasks_to_steal = list_size/2;
                }else{
                    tasks_to_steal=list_size;
                }
                old_tail = queue->tail;
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
                #pragma omp atomic write
                queue->list_size = (queue->list_size)-stolen_tasks;
                if(queue->list_size!=0){
                    helpNode->next->prev=NULL; //disconnect both
                    helpNode->next=NULL;
                    queue->tail=helpNode;
                }
            }
        }
        while(old_tail!=NULL){
            (* old_tail->task)(input_size); //execute task
            old_tail=old_tail->prev;
        }
    }

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
        //printf("%d\n", queue->list_size);
    }

    void init_queue(struct Queue* queue){
        queue->head = NULL;
        queue->tail = NULL;
        queue->next_queue = NULL;
        queue->prev_queue = NULL;
        queue->list_size=0;
    }

    void init_global_queue(struct Global_Queue* queue){
        queue->head_queue = NULL;
        queue->list_size=0;
    }

    //Original function taken from https://www.geeksforgeeks.org/doubly-linked-list/
    //This function prints contents of linked list starting from the given node
    void countListElements(struct Node* node){
        int count=0;
        while (node != NULL) {
            count++;
            node = node->next;
        }
        printf("\nElements:%d \n", count);
    }

    //Original function is taken from Parallel Computing (184.710)
    void fill_vector(double* m, int n) {
        for(int i=0; i<n; i++) {
            m[i] = 1.0;
        }
    }

    //Original function is taken from Parallel Computing (184.710)
    int free_vector(double* m) {
        if( m != NULL ) {
          free(m);
        }
        return 0;
    }

    //Original function is taken from Parallel Computing (184.710)
    void print_vector(double* m, int n, FILE *out) {
      for(int i=0; i<n; i++) {
          fprintf(out, "%8.4f ", m[i]);
      }
      printf("%d", omp_get_thread_num());
      fprintf(out, "\n");
    }

    //Matrix functions from (184.710)

    //Original function is taken from Parallel Computing (184.710)
    void fill_2d_matrix(double** m, int n) {
        for(int i=0; i<n; i++) {
          for(int j=0; j<n; j++) {
            m[i][j] = 1.0;
          }
        }
    }

    //Original function is taken from Parallel Computing (184.710) and http://c-faq.com/aryptr/dynmuldimary.html
    int create_2d_matrix(double*** m, int n) {
        *m = (double**)malloc(n * sizeof(double*));
        if(m == NULL) {
            printf("malloc failed - Not enough memory");
            exit(EXIT_FAILURE);
        } else {
            int i;
            for(i=0; i<n; i++) {
                double *array = (double*)malloc(n * n * sizeof(double));
                if(array == NULL) {
                    printf("malloc failed - Not enough memory");
                    exit(EXIT_FAILURE);
                }
                (*m)[i] = array;
            }
        }
        fill_2d_matrix(*m, n);
        return 0;
    }

    //Original function is taken from Parallel Computing (184.710)
    int free_2d_matrix(double** m) {
        if(m != NULL) {
          free(m[0]);
          free(m);
        }
        return 0;
    }

    //Original function is taken from Parallel Computing (184.710)
    void print_2d_matrix(double** m, int n, FILE *out) {
      for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
          fprintf(out, "%8.4f ", m[i][j]);
        }
        fprintf(out, "\n");
      }
    }

    //TESTS
    void testVectorVectorSum(double *C, int n){
        int count = 0;
        while (count<n){
            if(C[count]!=2){ //Correct calculation -> 1+1=2
                    printf("Wrong calculation of vectorVectorSum when using vectors initialized with 1 only!");
            }
            count++;
        }
        if(count!=n){ //Correct amount of elements - no data race
            printf("Wrong calculation of vectorVectorSum - some elements are missing: %d should be %d", count, n);
            exit(EXIT_FAILURE);
        }
       // printf("SIZE: %d\n", count );
    }

    void testMatrixVectorProduct(double *C, int n){
            int count = 0;
            while (count<n){
                if(C[count]!=n){ //Correct calculation? (n times +1 =n)
                        printf("Wrong calculation of matrixVectorProduct when using vector and matrix initialized with 1 only!");
                }
                count++;
            }
            if(count!=n){ //Correct amount of elements - no data race
                printf("Wrong calculation of matrixVectorProduct - some elements are missing: %d should be %d", count, n);
                exit(EXIT_FAILURE);
            }
           // printf("SIZE: %d\n", count );
    }

    void testMatrixMatrixProduct(double **C, int n){
            int count_row = 0;
            int count_column = 0;
            while (count_row<n){ //check if all n rows and columns have been calculated correctly
                count_column=0;
                while (count_column<n){
                    if(C[count_row][count_column]!=n){ //Correct calculation? (n times +1 =n)
                            printf("\nWrong calculation of matrixMatrixProduct when using matrices initialized with 1 only!");
                            exit(EXIT_FAILURE);
                    }
                    count_column++;
                }
                if(count_column!=n){ //Correct amount of elements - no data race
                    printf("Wrong calculation of matrixMatrixProduct - some column elements are missing: %d should be %d", count_column, n);
                    exit(EXIT_FAILURE);
                }
                count_row++;
            }
            if(count_row!=n){ //Correct amount of elements - no data race
                printf("Wrong calculation of matrixVectorProduct - some row elements are missing: %d should be %d", count_row, n);
                exit(EXIT_FAILURE);
            }
           // printf("SIZE: %d | %d\n", count_column, count_row);
    }

    //TASKS

    //Original function is taken from https://riptutorial.com/openmp/example/23425/addition-of-two-vectors-using-openmp-parallel-for-construct
    void vectorVectorSum (int n){
        double *A = (double*)malloc(n * sizeof(double));
        double *B = (double*)malloc(n * sizeof(double));
        double *C = (double*)malloc(n * sizeof(double));
        if( A == NULL || B == NULL || C == NULL) {
            printf("malloc failed - Not enough memory");
            exit(EXIT_FAILURE);
        }
        fill_vector(A, n);
        fill_vector(B, n);
        fill_vector(C, n);
        for (int i = 0; i < n; ++i){
            C[i] = A[i] + B[i];
        }
        double *test=C;
        //print_vector(C, n, stdout);
        testVectorVectorSum(test, n);
        free_vector(A);
        free_vector(B);
        free_vector(C);
    }

    //Original function is taken from https://www.appentra.com/parallel-computation-of-matrix-vector-product/
    void matrixVectorProduct (int n){
        double *V = (double*)malloc(n * sizeof(double));
        fill_vector(V, n);
        double *result = (double*)malloc(n * sizeof(double));
        fill_vector(result, n);
        if(V == NULL || result == NULL) {
            printf("malloc failed - Not enough memory");
            exit(EXIT_FAILURE);
        }
        double **M;
        create_2d_matrix(&M, n);
        for(int i=0;i<n;i++) {
          result[i]=0;
          for(int j=0;j<n;j++) {
              result[i] += M[i][j]*V[j];
          }
        }
//        print_vector(result, n, stdout);
        double *test=result;
        testMatrixVectorProduct(test, n);
        free_vector(V);
        free_2d_matrix(M);
        free_vector(test);
    }

    //Original function is taken from https://www.appentra.com/parallel-matrix-matrix-multiplication/
    void matrixMatrixProduct(int n){
        double **M1;
        create_2d_matrix(&M1, n);
        double **M2;
        create_2d_matrix(&M2, n);
        double **result;
        create_2d_matrix(&result, n);
        for (int i=0; i<n; i=i+1){
            for (int j=0; j<n; j=j+1){
                 result[i][j]=0.;
                 for (int k=0; k<n; k=k+1){
                    result[i][j]+=(M1[i][k])*(M2[k][j]);
                 }
            }
        }
        //print_2d_matrix(result, n, stdout);
        double **test=result;
        testMatrixMatrixProduct(test, n);
        free_2d_matrix(M1);
        free_2d_matrix(M2);
        free_2d_matrix(result);

    }

    void testQueueMemory(struct Queue* queue){
        if( queue == NULL ) {
                printf("malloc failed - Not enough memory");
                exit(EXIT_FAILURE);
        }
    }

    void pattern1WithoutWorkStealing(struct Queue* queue, void (*task_func_ptr)(int), int type, int tasks, int input_size, int p, int e){
        struct Node* head_next;
        #pragma omp parallel num_threads(p)
        {
             #pragma omp master //first add tasks into (global) queue
            {
                for(int i=0; i<tasks; i++){
                    push(queue, task_func_ptr);
                }
                head_next = queue->head;
            }
            #pragma omp barrier //wait for master to finish before executing the tasks
            #pragma omp single //Test if all tasks have been added to queue - implicit barrier, so no task can be executed before this test
            {
                if(queue->list_size!=tasks){
                    printf("Not all tasks have been added correctly in pattern1: %d should be %d", queue->list_size, tasks);
                    exit(EXIT_FAILURE);
                }//else printf("CORRECT: %d should be %d\n", queue->list_size, tasks);
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
                (* currentNode->task)(input_size); //execute task
                free(currentNode);
            }
        }
    }


//global queue
    void pattern1(struct Queue* queue, void (*task_func_ptr)(int), int type, int tasks, int input_size, int p, int e){
        if(e==1){ //No work stealing
            pattern1WithoutWorkStealing(queue, task_func_ptr, type, tasks, input_size, p, e);
        }else if (e==2){
            //work stealing
            pattern1WithoutWorkStealing(queue, task_func_ptr, type, tasks, input_size, p, e);
        }else{
            exit(EXIT_FAILURE);
        }

    }

    void executeTasks(struct Queue* queue, int iteration_end, int input_size){
        struct Node* head_next = queue->head;
        for (int i=0;i<iteration_end;i++){
            struct Node* currentNode;
            currentNode = head_next;
            head_next = head_next->next;
            (* currentNode->task)(input_size); //execute task
            queue->list_size=queue->list_size-1;
            free(currentNode); //Problem globale queue - pointer auf letzten setzen
        }
    }

    void pattern2WithoutWorkStealing(struct Global_Queue* global_queue, void (*task_func_ptr)(int), int type, int tasks, int input_size, int p, int e){
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
            for(int i=0; i<iteration_end; i++){ // add tasks into local queues
                push(local_queue, task_func_ptr);
            }
            if(local_queue->list_size!=iteration_end){
                printf("Not all tasks have been added correctly in local_queue pattern2: %d should be %d \n", local_queue->list_size, iteration_end);
                exit(EXIT_FAILURE);
            }
            executeTasks(local_queue, iteration_end, input_size);
            free(local_queue);
        }
    }

//ansatz 2 queues eine von der ich stehlen kann und eine normale? - Ende speichern, dass in die globale Queue - critical
    void pattern2WithWorkStealing(struct Global_Queue* global_queue, void (*task_func_ptr)(int), int type, int tasks, int input_size, int p, int e, int steal_size){
       #pragma omp parallel num_threads(p)
       {
           int rank=omp_get_thread_num();
           struct Queue* local_queue = (struct Queue*)malloc(sizeof(struct Queue));
           testQueueMemory(local_queue);
           init_queue(local_queue);
           #pragma omp for ordered  //add local queues into global one
               for (int i=0; i<p; i++) {
                   #pragma omp ordered
                   {  //printf("Rank: %d\n", rank);
                      pushQueue(global_queue, local_queue);
                      local_queue->index=i;
                   }
               }
           if(global_queue->list_size!=p){
               printf("Not all tasks have been added correctly in global_queue pattern2: %d should be %d\n", global_queue->list_size, p);
               exit(EXIT_FAILURE);
           }
           int iteration_end = (tasks%p)>rank ? (tasks/p)+1 : tasks/p;
           for(int i=0; i<iteration_end; i++){ // add tasks into local queues
               pushWithLock(local_queue, task_func_ptr);
           }
           #pragma omp atomic write
               local_queue->head->lock=0;
           if(local_queue->list_size !=iteration_end){
               printf("Not all tasks have been added correctly in local_queue pattern2: %d should be %d \n", local_queue->list_size, iteration_end);
               exit(EXIT_FAILURE);
           }
           //start executing and stealing
           while(local_queue->list_size>0){
                removeTailWithLock(local_queue, input_size);
           }
           //STEAL OTHER TASKS - from next
           if(local_queue==global_queue->head_queue){
                while(global_queue->tail_queue->list_size>1){
                   printf("STEALING prev: %d\n", omp_get_thread_num());
                    removeTailWithLock(global_queue->tail_queue, input_size);
                }
           }else{
               while(local_queue->prev_queue->list_size>1){
                    printf("STEALING prev: %d\n", omp_get_thread_num());
                    removeTailWithLock(local_queue->prev_queue, input_size);
               }
           } //steal from prev
           if(local_queue==global_queue->tail_queue){
               while(global_queue->head_queue->list_size>1){
                   printf("STEALING next: %d\n", omp_get_thread_num());
                   removeTailWithLock(global_queue->head_queue, input_size);
               }
           }else{
              while(local_queue->next_queue->list_size>1){
                    printf("STEALING next: %d\n", omp_get_thread_num());
                    removeTailWithLock(local_queue->next_queue, input_size);
              }
           }
          // free(local_queue);
        }
    }

//local queues
    void pattern2(struct Global_Queue* global_queue, void (*task_func_ptr)(int), int type, int tasks, int input_size, int p, int e){
            if(e==1){ //No work stealing
                pattern2WithoutWorkStealing(global_queue, task_func_ptr, type, tasks, input_size, p, e);
            }else if (e==2){ //Random selection - select one - threashhold...
                pattern2WithWorkStealing(global_queue, task_func_ptr, type, tasks, input_size, p, e, 10);
            }else{
                exit(EXIT_FAILURE);
            }
    }

    void pattern3(struct Queue* queue, void (*task_func_ptr)(int), int type, int tasks, int input_size, int p, int e){
        if (tasks < 1){
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
                (* currentNode->task)(input_size); //execute task
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
                            push(queue, task_func_ptr);
                            pattern3(queue, task_func_ptr, type, tasks, input_size, p, e);
                        }
                    }
                }
            }
        }
    }

    void execute_program(void (*task_func_ptr)(int), int create, int type, int m, int n, int execute, int p, int rep){
        double mean=0;
        for(int i=0; i<rep; i++){
            double start_time = omp_get_wtime();
            if(create==1){
                struct Queue* queue = (struct Queue*)malloc(sizeof(struct Queue));
                testQueueMemory(queue);
                init_queue(queue);
                pattern1 (queue, task_func_ptr, type, m, n, p, execute);
                free(queue);
            }else if(create==2){
                struct Global_Queue* queue = (struct Global_Queue*)malloc(sizeof(struct Global_Queue));
                if( queue == NULL ) {
                    printf("malloc failed - Not enough memory");
                    exit(EXIT_FAILURE);
                }
                init_global_queue(queue);
                pattern2 (queue, task_func_ptr, type, m, n, p, execute);
                free(queue);
            }else if(create==3){
               // pattern3 (queue, task_func_ptr, type, m, n, p, execute);
            }else{
                exit(EXIT_FAILURE);
            }
            double time = omp_get_wtime() - start_time;
            //printf("%f\n", time);
            mean+=time;
        }
        printf("Done in an average time of: %f\n", (mean/rep));
    }

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
        //Create tasks here
       void (*task_func_ptr)(int);
        if(type==1){
            task_func_ptr =  &vectorVectorSum;
        }else if(type==2){
            task_func_ptr = &matrixVectorProduct;
        }else if(type==3){
            task_func_ptr = &matrixMatrixProduct;
        }else{
            exit(EXIT_FAILURE);
        }

        execute_program(task_func_ptr, create, type, m, n, execute, p, rep);
    }
