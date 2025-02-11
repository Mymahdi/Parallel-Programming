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
---

## **Conclusion**
The `takpaz.cpp` program effectively utilizes multi-threading to manage a bakery system. The implementation ensures efficient order processing, synchronized baking, and fair resource distribution among multiple customers. Additionally, it collects performance metrics for evaluation.

# Report on `chandpaz.cpp`

## Overview
The `chandpaz.cpp` program is a modified version of `takpaz.cpp`, designed to handle a bakery system where multiple customers place orders, a baker processes these orders, and an oven bakes the bread. While similar in structure to `takpaz.cpp`, `chandpaz.cpp` introduces differences in algorithmic execution and synchronization mechanisms.

---

## Algorithm Breakdown

### 1. **Global Variables and Constants**
- `OVEN_CAPACITY = 10`: Maximum number of breads baked at once.
- `BAKING_TIME = 2000ms`: Time required to bake a batch.
- `customerQueue`: Stores customer names and order sizes.
- `sharedSpace`: Tracks baked bread available for customers.
- `queueMutex`, `sharedSpaceMutex`: Mutexes for thread safety.
- `cvBaker`, `cvOven`, `cvCustomer`: Condition variables for synchronization.
- `bakeryOpen`: Flag for bakery status.
- `ovenReady`: Flag for oven availability.
- `orderTimes`, `receiveTimes`: Vectors tracking performance metrics.

---

### 2. **Oven Function (`oven`)**
- Waits for the baker to initiate baking.
- Sleeps for `BAKING_TIME` to simulate baking.
- Notifies the baker when baking is complete.
- Continues until bakery closure.

---

### 3. **Baker Function (`baker`)**
- Extracts orders from `customerQueue`.
- Processes orders in batches up to `OVEN_CAPACITY`.
- Uses `cvOven` to coordinate baking.
- Moves baked bread to `sharedSpace`.
- Records processing times.

---

### 4. **Customer Function (`customer`)**
- Waits for bread in `sharedSpace`.
- Collects bread batch by batch.
- Updates `sharedSpace` and notifies baker.
- Records order fulfillment time.

---

### 5. **Input Handling (`getInput`)**
- Reads customer data and stores it in `customerQueue`.

---

### 6. **Performance Metrics (`calculateAndPrintMetrics`)**
- Computes average and standard deviation of:
  - Order processing time.
  - Order pickup time.
- Displays performance results.

---

## **Key Differences Between `takpaz.cpp` and `chandpaz.cpp`**
| Feature | `takpaz.cpp` | `chandpaz.cpp` |
|---------|-------------|-------------|
| **Baking Process** | Bakes continuously as orders arrive | May group multiple orders before baking |
| **Thread Coordination** | Customers pick up bread individually | Customers may collect in synchronized batches |
| **Synchronization Approach** | Uses strict condition variable checks | May allow flexible pickup timing |
| **Efficiency Consideration** | Processes each order as soon as possible | Optimizes batch processing to reduce wait times |

---

## **Concurrency and Synchronization**
- **Mutexes (`queueMutex`, `sharedSpaceMutex`)**: Protect shared resources.
- **Condition Variables (`cvBaker`, `cvOven`, `cvCustomer`)**:
  - `cvBaker`: Ensures baker works only when needed.
  - `cvOven`: Synchronizes oven baking cycles.
  - `cvCustomer`: Controls customer order collection.

---

## **Conclusion**
While `chandpaz.cpp` follows the same bakery system structure as `takpaz.cpp`, it introduces optimizations in order processing and pickup timing, making it more efficient in handling multiple customers concurrently.

---
---

# Report on `chandPaz-with-disorder.cpp`

## Overview
The `chandPaz-with-disorder.cpp` program simulates a bakery system with multiple bakers and customers, where customers place orders, bakers process them, and an oven is used to bake bread. The implementation uses **multi-threading**, **mutex locks**, and **condition variables** for synchronization, ensuring efficient resource handling. Unlike `chandpaz.cpp`, this implementation introduces **disordered processing**, where orders are not necessarily handled in a strict FIFO manner.

---

## Algorithm Breakdown

### 1. **Global Variables and Constants**
- `OVEN_CAPACITY`: Set dynamically based on the number of bakers.
- `BAKING_TIME = 2000ms`: Time required for a baking cycle.
- `bakerQueues`: Stores the queue of orders for each baker.
- `customerQueues`: Stores the customer orders assigned to bakers.
- `sharedSpace`: Tracks the number of breads available for each customer.
- `ovenMutex`, `sharedSpaceMutex`: Mutexes for synchronized access to shared resources.
- `cvOven`: Condition variable to manage oven availability.
- `bakeryOpen`: Flag indicating if the bakery is operational.
- `ovenReady`: Flag determining oven availability.
- `ovenCurrentCapacity`: Tracks the current load of the oven.

---

### 2. **Baker Function (`baker`)**
- Each baker thread waits for available oven capacity.
- Locks the oven mutex and waits on `cvOven` to ensure oven is not overloaded.
- Determines the batch size based on available capacity.
- Sleeps for `BAKING_TIME` to simulate baking.
- Updates `sharedSpace` with completed orders and notifies waiting customers.

#### **Difference from `chandpaz.cpp`**:
- `chandPaz-with-disorder.cpp` does not process orders strictly in sequence.
- `ovenCurrentCapacity` is dynamically adjusted instead of processing full batches sequentially.

---

### 3. **Customer Function (`customer`)**
- Customers place orders and wait until bread is available.
- Uses `cvOven.wait()` to ensure synchronization with baking.
- Picks up available bread batch by batch.
- Logs order completion time.

#### **Difference from `chandpaz.cpp`**:
- `chandpaz.cpp` customers wait for complete orders, while `chandPaz-with-disorder.cpp` allows partial pickups.
- Synchronization between customers and bakers is more dynamic, allowing overlap.

---

### 4. **Performance Metrics Calculation (`calculateAndPrintMetrics`)**
- Computes the **average** and **standard deviation** of:
  - Order processing time.
  - Bread receiving time.
- Prints these statistics for evaluation.

---

## **Key Differences from `chandpaz.cpp`**

| Feature | `chandpaz.cpp` | `chandPaz-with-disorder.cpp` |
|---------|---------------|----------------------------|
| **Order Processing** | Strict FIFO order handling | Disordered processing (more flexible) |
| **Baker Handling** | Bakers work in a fixed queue | Bakers process orders dynamically |
| **Oven Usage** | Fully filled batches | Flexible batch sizes |
| **Customer Pickup** | Customers receive full orders | Customers pick up partial orders |
| **Synchronization** | More structured | More flexible and dynamic |

---

## **Conclusion**
The `chandPaz-with-disorder.cpp` program introduces a more dynamic and flexible approach to order processing compared to `chandpaz.cpp`. Instead of enforcing a strict sequential order, it allows overlapping operations where customers can pick up bread in portions, and bakers adjust batch sizes dynamically. This approach optimizes oven usage but may introduce variability in order completion times.

While `chandpaz.cpp` ensures orderly execution, `chandPaz-with-disorder.cpp` prioritizes efficiency, reducing idle time for the bakers and oven. The choice between these two implementations depends on whether maintaining strict order or maximizing throughput is the priority.

