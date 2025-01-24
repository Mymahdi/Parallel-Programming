#include <iostream>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <cmath>
#include <sstream>

using namespace std;

const int OVEN_CAPACITY = 10;
const int BAKING_TIME = 2000;

vector<pair<string, int>> customerQueue;
unordered_map<string, int> sharedSpace;

mutex queueMutex;
mutex sharedSpaceMutex;
condition_variable cvBaker;
condition_variable cvOven;
condition_variable cvCustomer;

bool bakeryOpen = true;
bool ovenReady = true;

vector<int> orderTimes;
vector<int> receiveTimes;


void oven() {
    while (bakeryOpen) {
        unique_lock<mutex> lock(sharedSpaceMutex);
        cvOven.wait(lock, [] { return !ovenReady || !bakeryOpen; });
        if (!bakeryOpen) break;
        this_thread::sleep_for(chrono::milliseconds(BAKING_TIME));
        ovenReady = true;
        cvBaker.notify_one();
    }
}


void baker() {
    while (bakeryOpen) {
        unique_lock<mutex> lock(queueMutex);
        cvBaker.wait(lock, [] { return !customerQueue.empty() || !bakeryOpen; });
        if (!bakeryOpen) break;

        auto [customerName, orderSize] = customerQueue.front();
        customerQueue.erase(customerQueue.begin());
        lock.unlock();

        auto startBakeTime = chrono::steady_clock::now();
        cout << "Baker: Processing order for " << customerName << " (" << orderSize << " breads)\n";

        while (orderSize > 0) {
            int bakeSize = min(orderSize, OVEN_CAPACITY);
            {
                unique_lock<mutex> ovenLock(sharedSpaceMutex);
                ovenReady = false;
                cvOven.notify_one();
                cvBaker.wait(ovenLock, [] { return ovenReady; });
            }
            {
                lock_guard<mutex> spaceLock(sharedSpaceMutex);
                sharedSpace[customerName] += bakeSize;
            }
            orderSize -= bakeSize;
            cout << "Baker: Baked " << bakeSize << " breads for " << customerName << "\n";
        }

        auto endBakeTime = chrono::steady_clock::now();
        orderTimes.push_back(chrono::duration_cast<chrono::milliseconds>(endBakeTime - startBakeTime).count());
        cvCustomer.notify_all();
        unique_lock<mutex> spaceLock(sharedSpaceMutex);
        cvBaker.wait(spaceLock, [&customerName] { return sharedSpace[customerName] == 0; });
        cout << "Baker: Order for " << customerName << " is complete.\n";
    }
}
