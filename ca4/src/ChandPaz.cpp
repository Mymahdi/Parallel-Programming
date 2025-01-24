#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <cmath>
#include <sstream>

using namespace std;

int OVEN_CAPACITY;
const int BAKING_TIME = 2000;

vector<vector<pair<string, int>>> bakerQueues;
unordered_map<string, int> sharedSpace;

mutex ovenMutex;
mutex sharedSpaceMutex;
mutex customerMutex;
condition_variable cvOven;
vector<unique_ptr<mutex>> queueMutexes;
vector<unique_ptr<condition_variable>> cvBakers;
condition_variable cvCustomer;

bool bakeryOpen = true;
int ovenCurrentCapacity = 0;

vector<int> orderTimes;
vector<int> receiveTimes;


void getInput(int numBakers) {
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
    for (int i = 0; i < numBakers; i++) {
        string customerNames;
        getline(cin, customerNames);

        string orderSizes;
        getline(cin, orderSizes);

        stringstream nameStream(customerNames);
        stringstream sizeStream(orderSizes);

        string name;
        int orderSize;

        while (nameStream >> name && sizeStream >> orderSize) {
            bakerQueues[i].push_back({name, orderSize});
        }
    }
}

void baker(int id) {
    while (bakeryOpen) {
        unique_lock<mutex> queueLock(*queueMutexes[id]);
        cvBakers[id]->wait(queueLock, [id] { return !bakerQueues[id].empty() || !bakeryOpen; });
        if (!bakeryOpen) break;
        queueLock.unlock();

        auto [customerName, orderSize] = bakerQueues[id].front();
        bakerQueues[id].erase(bakerQueues[id].begin());
        auto startBakeTime = chrono::steady_clock::now();

        while (orderSize > 0) {
            int bakeSize;
            {
                unique_lock<mutex> queueLock(*queueMutexes[id]);
                bakeSize = min(orderSize, OVEN_CAPACITY - ovenCurrentCapacity);
                if (bakeSize == 0) {
                    cvOven.wait(queueLock, [] { return ovenCurrentCapacity < OVEN_CAPACITY; });
                    bakeSize = min(orderSize, OVEN_CAPACITY - ovenCurrentCapacity);
                }
                ovenCurrentCapacity += bakeSize;
                queueLock.unlock();
            }

            this_thread::sleep_for(chrono::milliseconds(BAKING_TIME));
            {
                lock_guard<mutex> spaceLock(sharedSpaceMutex);
                sharedSpace[customerName] += bakeSize;
                ovenCurrentCapacity -= bakeSize;
            }
            cvOven.notify_all();
            orderSize -= bakeSize;
        }

        auto endBakeTime = chrono::steady_clock::now();
        orderTimes.push_back(chrono::duration_cast<chrono::milliseconds>(endBakeTime - startBakeTime).count());
        cvCustomer.notify_all();
        unique_lock<mutex> queueRe2Lock(*queueMutexes[id]);
        cvBakers[id]->wait(queueRe2Lock, [&customerName] { return sharedSpace[customerName] == 0; });
        cout << "Baker " << id << ": Order for " << customerName << " is complete.\n";
        queueRe2Lock.unlock();
    }
}

void customer(const string &name, int orderSize, int bakerId) {
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
        cvBakers[bakerId]->notify_one();
    }

    auto endReceiveTime = chrono::steady_clock::now();
    unique_lock<mutex> customerLock(customerMutex);
    receiveTimes.push_back(chrono::duration_cast<chrono::milliseconds>(endReceiveTime - startReceiveTime).count());
    cout << "Customer " << name << " has received all breads and is leaving.\n";
    customerLock.unlock();
}

int main() {
    int numBakers;
    cin >> numBakers;

    OVEN_CAPACITY = 10 * numBakers;

    bakerQueues.resize(numBakers);
    queueMutexes.resize(numBakers);
    cvBakers.resize(numBakers);
    for (int i = 0; i < numBakers; ++i) {
        queueMutexes[i] = make_unique<mutex>();
        cvBakers[i] = make_unique<condition_variable>();
    }

    vector<thread> customerThreads;
    vector<thread> bakerThreads;

    getInput(numBakers);

    for (int i = 0; i < numBakers; ++i) {
        for (int j = 0; j < bakerQueues[i].size(); j++) {
            auto [name, orderSize] = bakerQueues[i][j];
            customerThreads.emplace_back(customer, name, orderSize, i);
        }
    }

    for (int i = 0; i < numBakers; ++i) {
        bakerThreads.emplace_back(baker, i);
    }

    for (auto &t : customerThreads) {
        if (t.joinable()) t.join();
    }

    bakeryOpen = false;
    for (auto &cv : cvBakers) {
        cv->notify_all();
    }
    for (auto &t : bakerThreads) {
        if (t.joinable()) t.join();
    }

    calculateAndPrintMetrics();
    cout << "Bakery is now closed.\n";
    return 0;
}


