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


void customer(const string& name, int orderSize) {
    auto startReceiveTime = chrono::steady_clock::now();

    while (orderSize > 0) {
        unique_lock<mutex> lock(sharedSpaceMutex);
        cvCustomer.wait(lock, [&name] { return sharedSpace[name] > 0; });

        int pickedUp = min(orderSize, sharedSpace[name]);
        orderSize -= pickedUp;
        sharedSpace[name] -= pickedUp;
        cout << "Customer " << name << " picked up " << pickedUp << " breads. Remaining order: " << orderSize << "\n";

        if (orderSize == 0) {
            sharedSpace.erase(name);
        }
        cvBaker.notify_one();
    }

    auto endReceiveTime = chrono::steady_clock::now();
    receiveTimes.push_back(chrono::duration_cast<chrono::milliseconds>(endReceiveTime - startReceiveTime).count());
    cout << "Customer " << name << " has received all breads and is leaving.\n";
}

void getInput() {
    string customerNames;
    getline(cin, customerNames);
    string orderSizes;
    getline(cin, orderSizes);

    stringstream nameStream(customerNames);
    stringstream sizeStream(orderSizes);

    string name;
    int orderSize;

    while (getline(nameStream, name, ' ') && sizeStream >> orderSize) {
        if (!name.empty()) {
            lock_guard<mutex> lock(queueMutex);
            customerQueue.push_back({name, orderSize});
        }
    }
}

void calculateAndPrintMetrics() {
    double avgOrder = 0.0, avgReceive = 0.0;
    for (auto time : orderTimes) avgOrder += time;
    for (auto time : receiveTimes) avgReceive += time;

    avgOrder /= orderTimes.size();
    avgReceive /= receiveTimes.size();

    double sdOrder = 0.0, sdReceive = 0.0;
    for (auto time : orderTimes) sdOrder += pow(time - avgOrder, 2);
    for (auto time : receiveTimes) sdReceive += pow(time - avgReceive, 2);

    sdOrder = sqrt(sdOrder / orderTimes.size());
    sdReceive = sqrt(sdReceive / receiveTimes.size());

    cout << "Average order processing time: " << avgOrder << " ms\n";
    cout << "Standard deviation of order processing time: " << sdOrder << " ms\n";
    cout << "Average bread receiving time: " << avgReceive << " ms\n";
    cout << "Standard deviation of bread receiving time: " << sdReceive << " ms\n";
}

int main() {
    vector<thread> customerThreads;
    getInput();

    for (int i = 0; i < customerQueue.size(); i++) {
        auto [name, orderSize] = customerQueue[i];
        customerThreads.emplace_back(customer, name, orderSize);
    }

    thread bakerThread(baker);
    thread ovenThread(oven);

    for (auto& t : customerThreads) {
        t.join();
    }

    bakeryOpen = false;
    cvBaker.notify_all();
    cvOven.notify_all();
    bakerThread.join();
    ovenThread.join();

    calculateAndPrintMetrics();

    return 0;
}

