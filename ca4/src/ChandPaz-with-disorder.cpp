#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <cmath>
#include <sstream>

using namespace std;

// Constants and global variables
int OVEN_CAPACITY;
const int BAKING_TIME = 2000;

vector<vector<pair<string, int>>> bakerQueues;
vector<vector<pair<string, int>>> customerQueues;
unordered_map<string, int> sharedSpace;

mutex ovenMutex;
mutex sharedSpaceMutex;
condition_variable cvOven;

bool bakeryOpen = true;
bool ovenReady = true;
int ovenCurrentCapacity = 0;

// Forward declarations
void getInput(int numBakers);
void calculateAndPrintMetrics();

void baker(int id) {
    while (bakeryOpen) {
        unique_lock<mutex> queueLock(ovenMutex);

        cvOven.wait(queueLock, [] { return ovenCurrentCapacity < OVEN_CAPACITY || !bakeryOpen; });
        if (!bakeryOpen) break;

        // Simulate baking process
        int bakeSize = min(10, OVEN_CAPACITY - ovenCurrentCapacity);
        ovenCurrentCapacity += bakeSize;

        queueLock.unlock();
        this_thread::sleep_for(chrono::milliseconds(BAKING_TIME));

        // Update shared space
        lock_guard<mutex> spaceLock(sharedSpaceMutex);
        ovenCurrentCapacity -= bakeSize;

        cvOven.notify_all();
    }
}
