
/*
 * Copyright (C) 2017 GreenWaves Technologies
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 */


/* Autotiler includes. */
#include "tdm_xxs_single_ce_224_best.h"
#include "tdm_xxs_single_ce_224_bestKernels.h"
#include "gaplib/fs_switch.h"
#include "measurments_utils.h"



#ifndef STACK_SIZE
#define STACK_SIZE      1024
#endif



AT_DEFAULTFLASH_EXT_ADDR_TYPE tdm_xxs_single_ce_224_best_L3_Flash = 0;
AT_DEFAULTFLASH_EXT_ADDR_TYPE tdm_xxs_single_ce_224_best_L3_PrivilegedFlash = 0;




/* Inputs */
L2_MEM unsigned char Input_1[150528];
L2_MEM unsigned char Input_2[1];
/* Outputs */
L2_MEM unsigned char Output_1[19];

/* Copy inputs functions */
switch_fs_t input_fs;
void *Input_File_Input_1;
int Input_File_Input_1_Position;
void *Input_File_Input_2;
int Input_File_Input_2_Position;
#define EXPECTED_NUM_ITERATIONS 1
int open_inputs() {
    __FS_INIT(input_fs);

    /* opening file Input_1 */
    #ifdef __EMUL__
    Input_File_Input_1 = __OPEN_READ(input_fs, "../Input_1.bin");
    #else
    Input_File_Input_1 = __OPEN_READ(input_fs, "../Input_1.bin");
    #endif
    if (!Input_File_Input_1) return 1;
    Input_File_Input_1_Position = 0;
    /* opening file Input_2 */
    #ifdef __EMUL__
    Input_File_Input_2 = __OPEN_READ(input_fs, "../Input_2.bin");
    #else
    Input_File_Input_2 = __OPEN_READ(input_fs, "../Input_2.bin");
    #endif
    if (!Input_File_Input_2) return 1;
    Input_File_Input_2_Position = 0;
    return 0;
}

int copy_inputs(int num_iterations) {
    {
        /* Reading from file Input_1 */
        int ret_Input_1 = 0;
        __SEEK(Input_File_Input_1, Input_File_Input_1_Position);
        ret_Input_1 = __READ(Input_File_Input_1, Input_1, 150528);
        if (ret_Input_1 != 150528) {
            return 0;
        }
        Input_File_Input_1_Position = 0;
    }
    {
        /* Reading from file Input_2 */
        int ret_Input_2 = 0;
        __SEEK(Input_File_Input_2, Input_File_Input_2_Position);
        ret_Input_2 = __READ(Input_File_Input_2, Input_2, 1);
        if (ret_Input_2 != 1) {
            return 0;
        }
        Input_File_Input_2_Position = 0;
    }
    return 1;
}

void close_inputs() {
    __CLOSE(Input_File_Input_1);

    __CLOSE(Input_File_Input_2);

    __FS_DEINIT(input_fs);
}

/* Copy outputs functions */
switch_fs_t output_fs;
int open_outputs() {
    return 0;
}

void write_outputs() {
}

void close_outputs() {
}





static void cluster(void * arg)
{
    #ifdef PERF
    printf("Start timer\n");
    gap_cl_starttimer();
    gap_cl_resethwtimer();
    #endif
    int iteration = (int) arg;
    tdm_xxs_single_ce_224_bestCNN_ConstructCluster();
GPIO_HIGH();
    tdm_xxs_single_ce_224_bestCNN(Input_1, Input_2, Output_1);
GPIO_LOW();
    printf("Runner completed: %d\n", iteration);

    int pred_idx = 0;
    int pred_val = ((unsigned char *)Output_1)[0];

    printf("Output logits/uint8:");
    for (int i = 0; i < 19; i++) {
        int v = ((unsigned char *)Output_1)[i];
        printf(" %d", v);
        if (v > pred_val) {
            pred_val = v;
            pred_idx = i;
        }
    }
    printf("\n");
    printf("Predicted class index: %d, raw score: %d\n", pred_idx, pred_val);

}

int main(int argc, char *argv[])
{
    printf("\n\n\t *** NNTOOL tdm_xxs_single_ce_224_best Example ***\n\n");
    printf("Entering main controller\n");

    /* Configure And open cluster. */
    OPEN_GPIO_MEAS();
    struct pi_device cluster_dev;
    struct pi_cluster_conf cl_conf;
    pi_cluster_conf_init(&cl_conf);
    cl_conf.cc_stack_size = STACK_SIZE;

    cl_conf.id = 0; /* Set cluster ID. */
                    // Enable the special icache for the master core
    cl_conf.icache_conf = PI_CLUSTER_MASTER_CORE_ICACHE_ENABLE |
                    // Enable the prefetch for all the cores, it's a 9bits mask (from bit 2 to bit 10), each bit correspond to 1 core
                    PI_CLUSTER_ICACHE_PREFETCH_ENABLE |
                    // Enable the icache for all the cores
                    PI_CLUSTER_ICACHE_ENABLE;

    pi_open_from_conf(&cluster_dev, (void *) &cl_conf);
    if (pi_cluster_open(&cluster_dev))
    {
        printf("Cluster open failed !\n");
        return -4;
    }

    /* Frequency Settings: defined in the Makefile */
    int cur_fc_freq = pi_freq_set(PI_FREQ_DOMAIN_FC, FREQ_FC*1000*1000);
    int cur_cl_freq = pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_CL*1000*1000);
    int cur_pe_freq = pi_freq_set(PI_FREQ_DOMAIN_PERIPH, FREQ_PE*1000*1000);
    if (cur_fc_freq == -1 || cur_cl_freq == -1 || cur_pe_freq == -1)
    {
        printf("Error changing frequency !\nTest failed...\n");
        return -4;
    }
	printf("FC Frequency = %d Hz CL Frequency = %d Hz PERIPH Frequency = %d Hz\n", 
            pi_freq_get(PI_FREQ_DOMAIN_FC), pi_freq_get(PI_FREQ_DOMAIN_CL), pi_freq_get(PI_FREQ_DOMAIN_PERIPH));

	#ifdef VOLTAGE
	pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, VOLTAGE);
	pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, VOLTAGE);
	#endif
	printf("Voltage: %dmV\n", pi_pmu_voltage_get(PI_PMU_VOLTAGE_DOMAIN_CHIP));

    


    
    

    // IMPORTANT - MUST BE CALLED AFTER THE CLUSTER IS SWITCHED ON!!!!
    printf("Constructor\n");
    int ConstructorErr = tdm_xxs_single_ce_224_bestCNN_Construct();
    if (ConstructorErr)
    {
        printf("Graph constructor exited with error: (%s)\n", GetAtErrorName(ConstructorErr));
        return -6;
    }
    
    


    /*
     * Put here Your input settings
     */
    if (open_inputs()) return -7;
    if (open_outputs()) return -8;
    printf("Call cluster\n");

    struct pi_cluster_task task;
    pi_cluster_task(&task, (void (*)(void *))cluster, NULL);
    pi_cluster_task_stacks(&task, NULL, SLAVE_STACK_SIZE);


    int iteration = 0;
    while (iteration < EXPECTED_NUM_ITERATIONS) {
        if (EXPECTED_NUM_ITERATIONS > 1) printf("Iteration: %d of %d\n", iteration, EXPECTED_NUM_ITERATIONS);
        if (!copy_inputs(iteration)) return -9;

        task.arg = (void *)iteration;
        pi_cluster_send_task_to_cl(&cluster_dev, &task);

        write_outputs();
        iteration++;
    }
    close_inputs();
    close_outputs();
    

    tdm_xxs_single_ce_224_bestCNN_Destruct();

#ifdef PERF
	{
		unsigned long long int TotalCycles = 0, TotalOper = 0;
		printf("\n");
		for (unsigned int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
			TotalCycles += AT_GraphPerf[i]; TotalOper += AT_GraphOperInfosNames[i];
		}
		for (unsigned int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
			printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", AT_GraphNodeNames[i], AT_GraphPerf[i], 100*((float) (AT_GraphPerf[i]) / TotalCycles), AT_GraphOperInfosNames[i], 100*((float) (AT_GraphOperInfosNames[i]) / TotalOper), ((float) AT_GraphOperInfosNames[i])/ AT_GraphPerf[i]);
		}
		printf("\n");
		printf("%45s: Cycles: %12llu, Cyc%%: 100.0%%, Operations: %12llu, Op%%: 100.0%%, Operations/Cycle: %f\n", "Total", TotalCycles, TotalOper, ((float) TotalOper)/ TotalCycles);
		printf("\n");
	}
#endif

    printf("Ended\n");
    return 0;
}
