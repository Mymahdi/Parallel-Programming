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
