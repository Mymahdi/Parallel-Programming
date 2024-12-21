#include <iostream>
#include <queue>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <sstream>
#include <string>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace chrono;

struct Order {
    string customerName;
    int breadCount;
};

// Get the current timestamp as a string
string getCurrentTime() {
    auto now = system_clock::now();
    auto timeT = system_clock::to_time_t(now);
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

    stringstream ss;
    ss << put_time(localtime(&timeT), "%H:%M:%S") << "." << setw(3) << setfill('0') << ms.count();
    return ss.str();
}

class Bakery {
private:
    int bakerCount;
    int ovenCapacity;
    vector<queue<Order>> queues;
    queue<Order> deliverySpace;
    mutex ovenMutex, deliveryMutex, queueMutex;
    condition_variable ovenCV, deliveryCV;
    int currentOvenCapacity;

    void baker(int bakerID) {
        while (true) {
            Order order;
            {
                unique_lock<mutex> lock(queueMutex);
                if (queues[bakerID].empty()) {
                    break;
                }
                order = queues[bakerID].front();
                queues[bakerID].pop();
                cout << "[" << getCurrentTime() << "] Baker " << bakerID + 1
                     << " starts processing order for " << order.customerName << " (" << order.breadCount << " bread)." << endl;
            }

            // Baking bread
            {
                unique_lock<mutex> lock(ovenMutex);
                ovenCV.wait(lock, [this, &order]() {
                    return currentOvenCapacity + order.breadCount <= ovenCapacity;
                });
                currentOvenCapacity += order.breadCount;
                cout << "[" << getCurrentTime() << "] Baker " << bakerID + 1
                     << " starts baking " << order.breadCount << " bread for " << order.customerName << "." << endl;
            }

            this_thread::sleep_for(chrono::seconds(2)); // Simulate baking time

            {
                lock_guard<mutex> lock(ovenMutex);
                currentOvenCapacity -= order.breadCount;
                ovenCV.notify_all();
                cout << "[" << getCurrentTime() << "] Baker " << bakerID + 1
                     << " finishes baking for " << order.customerName << "." << endl;
            }

            // Delivering bread
            {
                lock_guard<mutex> lock(deliveryMutex);
                for (int i = 0; i < order.breadCount; ++i) {
                    deliverySpace.push(order);
                }
                cout << "[" << getCurrentTime() << "] Baker " << bakerID + 1
                     << " delivers " << order.breadCount << " bread for " << order.customerName << "." << endl;
                deliveryCV.notify_all();
            }
        }
    }

public:
    Bakery(int bakers) : bakerCount(bakers), ovenCapacity(bakers * 10), currentOvenCapacity(0) {
        queues.resize(bakerCount);
    }

    void addOrders(int bakerID, const vector<Order>& orders) {
        for (const auto& order : orders) {
            queues[bakerID].push(order);
        }
    }

    void simulate() {
        vector<thread> bakers;
        for (int i = 0; i < bakerCount; ++i) {
            bakers.emplace_back(&Bakery::baker, this, i);
        }
        for (auto& t : bakers) {
            t.join();
        }
    }
};

int main() {
    int bakerCount;
    cin >> bakerCount;
    cin.ignore();

    Bakery bakery(bakerCount);
    for (int i = 0; i < bakerCount; ++i) {
        string names, counts;
        getline(cin, names);
        getline(cin, counts);

        vector<Order> orders;
        istringstream nameStream(names), countStream(counts);
        string name;
        int count;
        while (nameStream >> name && countStream >> count) {
            orders.push_back({name, count});
        }

        bakery.addOrders(i, orders);
    }

    bakery.simulate();
    return 0;
}
