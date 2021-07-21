#include "main_functions.h"

#include "dma_hardware_mover.h"
#include "conv_hardware.h"
#include "xil_cache.h"
#include "miscellaneous.h"
#include "event.h"

Event * interpreter_event_ = nullptr;
Event * conv_sw_event_ = nullptr;
Event * conv_hw_event_ = nullptr;
int conv_flag_;

Event * dma_flush_event_ = nullptr;
Event * dma_fetch_event_ = nullptr;
Event * dma_tx_event_ = nullptr;
Event * dma_rx_event_ = nullptr;
int flag_;

inline static void Accelerator_txISR (void * dma)
{
  ASSERT(dma != NULL);

  if (dma != NULL)
  {
    flag_ = 1;
  }
}

#define ACCELERATOR_DMA_RESET_TIMEOUT 1000

static void Accelerator_txInterruptHandler (void * data)
{
  void *     dma        = data;
  DMAIRQMask irq_status = DMAHardware_mover.InterruptGetStatus (dma, MEMORY_TO_HARDWARE);
  Event_stop (dma_tx_event_);

  DMAHardware_mover.InterruptClear (dma, irq_status, MEMORY_TO_HARDWARE);

  if (!(irq_status & DMA_IRQ_ALL)) return;

  if (irq_status & DMA_IRQ_DELAY) return;

  if (irq_status & DMA_IRQ_ERROR)
  {
    int TimeOut;

    DMAHardware_mover.Reset (dma);

    for (TimeOut = ACCELERATOR_DMA_RESET_TIMEOUT; 0 < TimeOut; TimeOut--)
      if (DMAHardware_mover.ResetIsDone (dma)) break;

    ASSERT(0);
    return;
  }

  if (irq_status & DMA_IRQ_IOC)
    Accelerator_txISR (dma);
}


inline static void Accelerator_rxISR (void * data)
{
  ASSERT (data != NULL);

  if (data != NULL)
  {
    flag_ = 2;
    //Xil_DCacheInvalidateRange ((INTPTR) rxBuffer, rxBufferSize);
  }
}


static void Accelerator_rxInterruptHandler (void * data)
{
  void *     dma         = data;
  DMAIRQMask irq_status  = DMAHardware_mover.InterruptGetStatus (dma, HARDWARE_TO_MEMORY);
  Event_stop (dma_rx_event_);

  DMAHardware_mover.InterruptClear (dma, irq_status, HARDWARE_TO_MEMORY);

  if (!(irq_status & DMA_IRQ_ALL)) return;

  if (irq_status & DMA_IRQ_DELAY) return;

  if (irq_status & DMA_IRQ_ERROR)
  {
    int TimeOut;

    DMAHardware_mover.Reset (dma);

    for (TimeOut = ACCELERATOR_DMA_RESET_TIMEOUT; 0 < TimeOut; TimeOut--)
      if (DMAHardware_mover.ResetIsDone (dma)) break;

    ASSERT(0);
    return;
  }

  if (irq_status & DMA_IRQ_IOC)
    Accelerator_rxISR (dma);
}

const void * baseAddress = (const void *) (XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x31000000);
const size_t buffer_size = 1024*1024;

int Dma_transaction (void * dmaHardware, void * tx_buffer, size_t tx_buffer_size,
                     void * rx_buffer, size_t rx_buffer_size)
{
  int status;

  flag_ = 0;

  if (tx_buffer != nullptr && 0 < tx_buffer_size)
  {
    Event_start (dma_flush_event_);
    Xil_DCacheFlushRange ((UINTPTR) tx_buffer, tx_buffer_size);
    Event_stop (dma_flush_event_);
    Event_start (dma_tx_event_);
    status = DMAHardware_mover.Move (dmaHardware, (void *) tx_buffer,
                                     tx_buffer_size, MEMORY_TO_HARDWARE);
  }

  if (rx_buffer != nullptr && 0 < rx_buffer_size)
  {
    Event_start (dma_fetch_event_);
    Event_start (dma_rx_event_);
    status = DMAHardware_mover.Move (dmaHardware, (void *) rx_buffer,
                                     rx_buffer_size, HARDWARE_TO_MEMORY);
    Xil_DCacheInvalidateRange ((INTPTR) rx_buffer, rx_buffer_size);
    Event_stop (dma_fetch_event_);
  }

  //while (!flag_);

  return status;
}

int Dma_initialize (void * dmaHardware, const int dma_device_id,
                    const int dmaTxIntVecID,
                    ARM_GIC_InterruptHandler txInterruptHandler,
                    const int dmaRxIntVecID,
                    ARM_GIC_InterruptHandler rxInterruptHandler)
{
  int status = DMAHardware_mover.Initialize (dmaHardware,
                                             dma_device_id);

  if (dmaTxIntVecID)
  {
    DMAHardware_mover.InterruptEnable (dmaHardware, DMA_IRQ_ALL,
                                       MEMORY_TO_HARDWARE);

    status = DMAHardware_mover.InterruptSetHandler (
        dmaHardware, dmaTxIntVecID, txInterruptHandler,
        dmaHardware);
    ASSERT(status == XST_SUCCESS);
  }

  if (dmaRxIntVecID)
  {
    DMAHardware_mover.InterruptEnable (dmaHardware, DMA_IRQ_ALL,
                                       HARDWARE_TO_MEMORY);

    status = DMAHardware_mover.InterruptSetHandler (
        dmaHardware, dmaRxIntVecID, rxInterruptHandler,
        dmaHardware);
    ASSERT(status == XST_SUCCESS);
    if (status != XST_SUCCESS)
      return status;
  }

  dma_flush_event_ = Event_new (interpreter_event_, EVENT_LAYER, (void *) "FLUSH");
  dma_tx_event_ = Event_new (dma_flush_event_, EVENT_HARDWARE, (void *) "TX_HW");

  dma_fetch_event_ = Event_new (interpreter_event_, EVENT_LAYER, (void *) "FETCH");
  dma_rx_event_ = Event_new (dma_fetch_event_, EVENT_HARDWARE, (void *) "RX_HW");

  return status;
}

int Platform_dma_loopback_test (void)
{
  void * dmaHardware = DMAHardware_mover.new_ ();
  int * rx_buffer = (int *) baseAddress;
  int * tx_buffer = (int *) (baseAddress + buffer_size);
  int status = XST_SUCCESS;

  for (size_t i = 0; i < (buffer_size / sizeof(int)); i++)
  {
    tx_buffer[i] = i;
    rx_buffer[i] = (buffer_size / sizeof(int)) - i;
  }

  status = Dma_initialize (dmaHardware, XPAR_AXI_DMA_0_DEVICE_ID,
                           XPAR_FABRIC_AXI_DMA_0_MM2S_INTROUT_INTR,
                           Accelerator_txInterruptHandler,
                           XPAR_FABRIC_AXI_DMA_0_S2MM_INTROUT_INTR,
                           Accelerator_rxInterruptHandler);

  status = Dma_transaction(dmaHardware, tx_buffer, buffer_size, rx_buffer, buffer_size);

  printf ("Test %s!", (memcmp (rx_buffer, tx_buffer, buffer_size) == 0) ? "PASSED" : "FAILED");

  return status;
}

static void Accelerator_hardwareInterruptHandler (void * data)
{
  void * conv = data;
  uint32_t status;

  Event_stop (conv_hw_event_);

  status = Conv_hardware.InterruptGetStatus (conv);
  Conv_hardware.InterruptClear (conv, status);

  /*!! Clear profile BEFORE making the accelerator ready !!*/
  conv_flag_ = status & 1;
}

int Conv_initialize (void * conv, const int conv_device_id, const int convIntVecID, ARM_GIC_InterruptHandler hardwareInterruptHandler)
{
  int status = Conv_hardware.Initialize (conv, conv_device_id);

  if (convIntVecID)
  {
    Conv_hardware.InterruptGlobalEnable (conv);
    Conv_hardware.InterruptEnable (conv, 1);
    Conv_hardware.InterruptSetHandler (conv, convIntVecID,
                                       hardwareInterruptHandler,
                                       conv);
  }

  interpreter_event_ = Event_new (nullptr, EVENT_NETWORK,
                                  (void*) "Interpreter");
  conv_sw_event_ = Event_new (interpreter_event_, EVENT_LAYER,
                              (void*) "CONV_2D_SW");
  conv_hw_event_ = Event_new (conv_sw_event_, EVENT_HARDWARE,
                              (void*) "CONV_2D_HW");
  return status;
}

int Conv_transaction (void)
{
  void * conv = Conv_hardware.new_ ();
  void * dmaHardware = DMAHardware_mover.new_ ();
  int status;

  status = Conv_initialize (conv, XPAR_CONV_0_DEVICE_ID,
                            XPAR_FABRIC_CONV_0_INTERRUPT_INTR,
                            Accelerator_hardwareInterruptHandler);

  status = Dma_initialize (dmaHardware, XPAR_AXI_DMA_0_DEVICE_ID,
                           XPAR_FABRIC_AXI_DMA_0_MM2S_INTROUT_INTR,
                           Accelerator_txInterruptHandler,
                           XPAR_FABRIC_AXI_DMA_0_S2MM_INTROUT_INTR,
                           Accelerator_rxInterruptHandler);

  status = Conv_hardware.Get_mode (conv);
  Conv_hardware.Set_mode (conv, 1);

  //Conv_hardware.Set_batches (conv, buffer_size / sizeof(int));
  Conv_hardware.Set_batches (conv, (buffer_size / sizeof(int)) / (2));

  status = Conv_hardware.Get_batches (conv);

  int * rx_buffer = (int *) baseAddress;
  int * tx_buffer = (int *) (baseAddress + buffer_size);

  for (size_t i = 0; i < (buffer_size / sizeof(int)); i++)
  {
    tx_buffer[i] = i;
  }

  memset (rx_buffer, 0, buffer_size);
  Xil_DCacheFlushRange ((UINTPTR) rx_buffer, buffer_size);

  while (!Conv_hardware.IsReady (conv));

  conv_flag_ = 0;

  Event_start (interpreter_event_);
  Event_start (conv_sw_event_);
  Event_start (conv_hw_event_);
  Conv_hardware.Start (conv);

  status = Dma_transaction (dmaHardware,
                            tx_buffer, buffer_size,
                            rx_buffer, buffer_size);

  while (!Conv_hardware.IsDone (conv));



  Event_stop (conv_sw_event_);

  Event_stop (interpreter_event_);

  Event_print (interpreter_event_);

  while (conv_flag_ < 1);

  return status;
}

int main (int argc, char* argv[])
{
  ARM_GIC_initialize ();

  Conv_transaction ();

  setup ();
  while (true)
  {
    loop ();
  }
}
