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

void getInput(int numBakers);
void calculateAndPrintMetrics();

void baker(int id) {
    while (bakeryOpen) {
        unique_lock<mutex> queueLock(ovenMutex);

        cvOven.wait(queueLock, [] { return ovenCurrentCapacity < OVEN_CAPACITY || !bakeryOpen; });
        if (!bakeryOpen) break;

        int bakeSize = min(10, OVEN_CAPACITY - ovenCurrentCapacity);
        ovenCurrentCapacity += bakeSize;

        queueLock.unlock();
        this_thread::sleep_for(chrono::milliseconds(BAKING_TIME));

        lock_guard<mutex> spaceLock(sharedSpaceMutex);
        ovenCurrentCapacity -= bakeSize;

        cvOven.notify_all();
    }
}

void customer(const string &name, int orderSize, int bakerId) {
    auto startReceiveTime = chrono::steady_clock::now();

    unique_lock<mutex> lock(sharedSpaceMutex);
    sharedSpace[name] += orderSize;

    cout << "Customer " << name << " placed an order for " << orderSize << " breads.\n";

    while (orderSize > 0) {
        cvOven.wait(lock, [&] { return sharedSpace[name] > 0; });

        int pickedUp = min(orderSize, sharedSpace[name]);
        orderSize -= pickedUp;
        sharedSpace[name] -= pickedUp;

        cout << "Customer " << name << " picked up " << pickedUp << " breads.\n";
    }

    auto endReceiveTime = chrono::steady_clock::now();
    cout << "Customer " << name << " has received all breads.\n";
}

void calculateAndPrintMetrics(vector<int> orderTimes, vector<int> receiveTimes) {
    double avgOrder = 0.0, avgReceive = 0.0;

    for (int time : orderTimes) avgOrder += time;
    for (int time : receiveTimes) avgReceive += time;

    avgOrder /= orderTimes.size();
    avgReceive /= receiveTimes.size();

    double sdOrder = 0.0, sdReceive = 0.0;
    for (int time : orderTimes) sdOrder += pow(time - avgOrder, 2);
    for (int time : receiveTimes) sdReceive += pow(time - avgReceive, 2);

    sdOrder = sqrt(sdOrder / orderTimes.size());
    sdReceive = sqrt(sdReceive / receiveTimes.size());

    cout << "Average order time: " << avgOrder << " ms\n";
    cout << "Order time standard deviation: " << sdOrder << " ms\n";
    cout << "Average receive time: " << avgReceive << " ms\n";
    cout << "Receive time standard deviation: " << sdReceive << " ms\n";
}

int main() {
    int numBakers;
    cin >> numBakers;

    OVEN_CAPACITY = 10 * numBakers;
    vector<thread> bakerThreads;
    vector<thread> customerThreads;

    getInput(numBakers);

    for (int i = 0; i < numBakers; ++i) {
        bakerThreads.emplace_back(baker, i);
    }

    for (int i = 0; i < customerQueues.size(); ++i) {
        for (auto &[name, orderSize] : customerQueues[i]) {
            customerThreads.emplace_back(customer, name, orderSize, i);
        }
    }

    for (auto &t : customerThreads) {
        if (t.joinable()) t.join();
    }

    bakeryOpen = false;
    cvOven.notify_all();

    for (auto &t : bakerThreads) {
        if (t.joinable()) t.join();
    }

    cout << "Bakery is now closed.\n";
    return 0;
}

