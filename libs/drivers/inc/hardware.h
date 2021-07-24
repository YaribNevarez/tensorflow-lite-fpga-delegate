/*
 * sbs_hardware.h
 *
 *  Created on: Mar 3rd, 2020
 *      Author: Yarib Nevarez
 */
#ifndef SBS_HARDWARE_H_
#define SBS_HARDWARE_H_

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#include <stdint.h>
#include <stddef.h>

#include "gic.h"

#include "xil_types.h"
#include "xstatus.h"
/***************** Macros (Inline Functions) Definitions *********************/


/**************************** Type Definitions *******************************/

typedef enum
{
  HW_INITIALIZE,
  HW_INFERENCE
} HardwareMode;

typedef struct
{
  void *    (*new_)(void);
  void      (*delete_)(void ** InstancePtr);

  int       (*Initialize) (void *InstancePtr, u16 deviceId);
  void      (*Start)      (void *InstancePtr);
  uint32_t  (*IsDone)     (void *InstancePtr);
  uint32_t  (*IsIdle)     (void *InstancePtr);
  uint32_t  (*IsReady)    (void *InstancePtr);
  void      (*EnableAutoRestart) (void *InstancePtr);
  void      (*DisableAutoRestart) (void *InstancePtr);
  uint32_t  (*Get_return) (void *InstancePtr);

  void      (*Set_batches) (void *InstancePtr, u32 Data);
  uint32_t  (*Get_batches) (void *InstancePtr);

  void      (*Set_input_height) (void *InstancePtr, u32 Data);
  uint32_t  (*Get_input_height) (void *InstancePtr);
  void      (*Set_input_width) (void *InstancePtr, u32 Data);
  uint32_t  (*Get_input_width) (void *InstancePtr);
  void      (*Set_input_depth) (void *InstancePtr, u32 Data);
  uint32_t  (*Get_input_depth) (void *InstancePtr);
  void      (*Set_filter_height) (void *InstancePtr, u32 Data);
  uint32_t  (*Get_filter_height) (void *InstancePtr);
  void      (*Set_filter_width) (void *InstancePtr, u32 Data);
  uint32_t  (*Get_filter_width) (void *InstancePtr);
  void      (*Set_output_height) (void *InstancePtr, u32 Data);
  uint32_t  (*Get_output_height) (void *InstancePtr);
  void      (*Set_output_width) (void *InstancePtr, u32 Data);
  uint32_t  (*Get_output_width) (void *InstancePtr);
  void      (*Set_output_depth) (void *InstancePtr, u32 Data);
  uint32_t  (*Get_output_depth) (void *InstancePtr);

  void      (*Set_mode)    (void *InstancePtr, uint32_t Data);
  uint32_t  (*Get_mode)    (void *InstancePtr);

  void      (*InterruptGlobalEnable)  (void *InstancePtr);
  void      (*InterruptGlobalDisable) (void *InstancePtr);
  void      (*InterruptEnable)        (void *InstancePtr, uint32_t Mask);
  void      (*InterruptDisable)       (void *InstancePtr, uint32_t Mask);
  void      (*InterruptClear)         (void *InstancePtr, uint32_t Mask);
  uint32_t  (*InterruptGetEnabled)    (void *InstancePtr);
  uint32_t  (*InterruptGetStatus)     (void *InstancePtr);

  uint32_t  (*InterruptSetHandler)    (void *InstancePtr,
                                       uint32_t ID,
                                       ARM_GIC_InterruptHandler handler,
                                       void * data);
} Hardware;

/************************** Constant Definitions *****************************/

/************************** Variable Definitions *****************************/

/************************** Function Prototypes ******************************/


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* SBS_HARDWARE_H_ */
