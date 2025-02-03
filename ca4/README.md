# Report on `takpaz.cpp`

## Overview
The `takpaz.cpp` program simulates a bakery system where multiple customers place orders for bread, a baker processes these orders, and an oven bakes the bread. The program employs **multi-threading** with **mutex locks** and **condition variables** to manage concurrent access to shared resources, ensuring proper synchronization.

---

## Algorithm Breakdown

### 1. **Global Variables and Constants**
- `OVEN_CAPACITY = 10`: Defines the maximum number of breads that can be baked at a time.
- `BAKING_TIME = 2000ms`: The time required to bake a batch of bread.
- `customerQueue`: A queue holding customer names and their respective order sizes.
- `sharedSpace`: A shared storage area that tracks the number of breads ready for each customer.
- `queueMutex`, `sharedSpaceMutex`: Mutexes to manage concurrent access to the order queue and shared space.
- `cvBaker`, `cvOven`, `cvCustomer`: Condition variables to coordinate tasks between customers, the baker, and the oven.
- `bakeryOpen`: A flag indicating whether the bakery is operational.
- `ovenReady`: A flag to determine when the oven is available.
- `orderTimes`, `receiveTimes`: Vectors storing order processing and receiving times for performance evaluation.

---

### 2. **Oven Function** (`oven`)
- Waits until the baker requests baking.
- Sleeps for `BAKING_TIME` milliseconds to simulate baking.
- Notifies the baker when the baking process is complete.
- Continues operation until the bakery closes.

---

### 3. **Baker Function** (`baker`)
- Waits for new customer orders in `customerQueue`.
- Picks up an order and starts processing.
- Bakes bread in batches of `OVEN_CAPACITY`.
- Uses condition variables to synchronize with the oven.
- Stores the completed breads in `sharedSpace` for pickup by customers.
- Records the time taken for order processing.

---

### 4. **Customer Function** (`customer`)
- Waits for their order to be available in `sharedSpace`.
- Picks up the bread batch by batch.
- Once the order is fulfilled, updates `sharedSpace` and notifies the baker.
- Records the time taken to receive the order.

---

### 5. **Input Handling** (`getInput`)
- Reads customer names and order sizes from standard input.
- Stores the input data into `customerQueue`.

---

### 6. **Performance Metrics Calculation** (`calculateAndPrintMetrics`)
- Computes the **average** and **standard deviation** of:
  - Order processing time.
  - Bread receiving time.
- Displays these statistics at the end of the program.

---

## **Main Function Execution Flow**
1. Reads input and fills `customerQueue`.
2. Spawns customer threads, each processing a customer's order.
3. Starts baker and oven threads to handle baking.
4. Waits for all customer orders to be fulfilled.
5. Closes the bakery and joins all threads.
6. Computes and prints performance metrics.

---

## **Concurrency and Synchronization**
- **Mutexes** ensure exclusive access to shared resources (`queueMutex`, `sharedSpaceMutex`).
- **Condition Variables** (`cvBaker`, `cvOven`, `cvCustomer`) synchronize tasks:
  - `cvBaker`: Ensures baker only works when there are orders.
  - `cvOven`: Ensures oven starts baking when required.
  - `cvCustomer`: Ensures customers pick up bread only when available.

---

## **Conclusion**
The `takpaz.cpp` program effectively utilizes multi-threading to manage a bakery system. The implementation ensures efficient order processing, synchronized baking, and fair resource distribution among multiple customers. Additionally, it collects performance metrics for evaluation.

