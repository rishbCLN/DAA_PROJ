#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std::chrono;
using namespace std;

static const int FEATURE_COUNT = 5;
static const int ALGO_COUNT = 5;
static const int KD_NEIGHBOR_COUNT = 5;
static const int MIN_GENERATED_ARRAYS = 10100;
static const int MIN_GENERATED_LENGTH = 7000;
static const int MAX_GENERATED_LENGTH = 100000;
static const long long MAX_GENERATED_VALUE = 1000000LL;
static const int TRAIN_BENCH_REPEATS = 3;
static const double VOTING_EPS = 1e-9;
static const double CLOSE_SCORE_RATIO = 0.92;

struct MemoryEntry {
    array<double, FEATURE_COUNT> features{};
    int size = 0;
    string algo;
    array<double, ALGO_COUNT> times{};
};

struct KDNode {
    int index = -1;
    int dim = 0;
    double split = 0.0;
    int left = -1;
    int right = -1;
};

static const array<string, ALGO_COUNT> ALGO_NAMES = {
    "InsertionSort",
    "CountingSort",
    "RadixSort",
    "QuickSort",
    "MergeSort"
};

static vector<MemoryEntry> gMemory;
static vector<array<double, FEATURE_COUNT>> gNormalizedPoints;
static vector<int> gWinningAlgoIndex;
static vector<KDNode> gKDNodes;
static array<double, FEATURE_COUNT> gFeatureMean{};
static array<double, FEATURE_COUNT> gFeatureStd{};
static int gKDRoot = -1;
static bool gModelReady = false;
static const string LEARNING_FILE = "learning.csv";
static const string NORMALIZATION_FILE = "learning_norm.csv";

static void insertionSort(vector<int> &a) {
    int n = (int)a.size();
    for (int i = 1; i < n; ++i) {
        int key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            --j;
        }
        a[j + 1] = key;
    }
}

static void merge(vector<int> &a, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;
    vector<int> left(n1), right(n2);
    for (int i = 0; i < n1; ++i) left[i] = a[l + i];
    for (int j = 0; j < n2; ++j) right[j] = a[m + 1 + j];

    int i = 0;
    int j = 0;
    int k = l;
    while (i < n1 && j < n2) {
        if (left[i] <= right[j]) {
            a[k++] = left[i++];
        } else {
            a[k++] = right[j++];
        }
    }
    while (i < n1) a[k++] = left[i++];
    while (j < n2) a[k++] = right[j++];
}

static void mergeSort(vector<int> &a, int l, int r) {
    if (l >= r) return;
    int m = l + (r - l) / 2;
    mergeSort(a, l, m);
    mergeSort(a, m + 1, r);
    merge(a, l, m, r);
}

static int medianOfThree(vector<int> &a, int low, int high) {
    int mid = (low + high) / 2;
    if (a[low] > a[mid]) swap(a[low], a[mid]);
    if (a[mid] > a[high]) swap(a[mid], a[high]);
    if (a[low] > a[mid]) swap(a[low], a[mid]);
    return a[mid];
}

static int partitionQs(vector<int> &a, int low, int high) {
    int pivot = medianOfThree(a, low, high);
    int i = low;
    int j = high;
    while (i <= j) {
        while (a[i] < pivot) ++i;
        while (a[j] > pivot) --j;
        if (i <= j) {
            swap(a[i], a[j]);
            ++i;
            --j;
        }
    }
    return i;
}

static void quickSort(vector<int> &a, int low, int high) {
    if (low >= high) return;
    int p = partitionQs(a, low, high);
    quickSort(a, low, p - 1);
    quickSort(a, p, high);
}

static void countingSort(vector<int> &a) {
    if (a.empty()) return;
    int n = (int)a.size();
    int minv = a[0];
    int maxv = a[0];
    for (int i = 0; i < n; ++i) {
        if (a[i] < minv) minv = a[i];
        if (a[i] > maxv) maxv = a[i];
    }

    int range = maxv - minv + 1;
    vector<int> count(range, 0), out(n);
    for (int i = 0; i < n; ++i) count[a[i] - minv]++;
    for (int i = 1; i < range; ++i) count[i] += count[i - 1];
    for (int i = n - 1; i >= 0; --i) {
        out[count[a[i] - minv] - 1] = a[i];
        --count[a[i] - minv];
    }
    a = out;
}

static int getMax(vector<int> &a) {
    int mx = a[0];
    for (int i = 1; i < (int)a.size(); ++i) {
        if (a[i] > mx) mx = a[i];
    }
    return mx;
}

static void radixSort(vector<int> &a) {
    if (a.empty()) return;
    int n = (int)a.size();
    int mx = getMax(a);

    for (int exp = 1; mx / exp > 0; exp *= 10) {
        vector<int> out(n);
        vector<int> count(10, 0);

        for (int i = 0; i < n; ++i) {
            int digit = (a[i] / exp) % 10;
            ++count[digit];
        }

        for (int i = 1; i < 10; ++i) count[i] += count[i - 1];
        for (int i = n - 1; i >= 0; --i) {
            int digit = (a[i] / exp) % 10;
            out[count[digit] - 1] = a[i];
            --count[digit];
        }
        a = out;
    }
}

static int algoIndexFromName(const string &algo) {
    for (int i = 0; i < ALGO_COUNT; ++i) {
        if (ALGO_NAMES[i] == algo) return i;
    }
    return -1;
}

static string algoNameFromIndex(int index) {
    if (index < 0 || index >= ALGO_COUNT) return "NONE";
    return ALGO_NAMES[index];
}

static double calcEntropy(const vector<int> &a) {
    int n = (int)a.size();
    if (n == 0) return 0.0;

    unordered_map<int, int> counts;
    for (int value : a) {
        counts[value]++;
    }

    double entropy = 0.0;
    for (const auto &entry : counts) {
        double p = (double)entry.second / (double)n;
        entropy += p * log2(1.0 / p);
    }
    return entropy;
}

static array<double, FEATURE_COUNT> extractFeatures(const vector<int> &a) {
    array<double, FEATURE_COUNT> features{};
    int n = (int)a.size();
    if (n == 0) {
        return features;
    }

    unordered_map<int, int> counts;
    int minv = a[0];
    int maxv = a[0];
    int nondecreasing = 0;

    for (int i = 0; i < n; ++i) {
        int value = a[i];
        counts[value]++;
        if (value < minv) minv = value;
        if (value > maxv) maxv = value;
        if (i > 0) {
            if (a[i] >= a[i - 1]) ++nondecreasing;
        }
    }

    features[0] = (double)n;
    features[1] = (n > 1) ? (double)nondecreasing / (double)(n - 1) : 1.0;
    features[2] = (double)counts.size() / (double)n;
    features[3] = (double)(maxv - minv) / (double)max(1, n);
    features[4] = calcEntropy(a);
    return features;
}

static bool allNonNegative(const vector<int> &a) {
    for (int value : a) {
        if (value < 0) return false;
    }
    return true;
}

static bool countingSortAllowed(const vector<int> &a) {
    if (a.empty()) return false;
    int minv = a[0];
    int maxv = a[0];
    for (int value : a) {
        if (value < minv) minv = value;
        if (value > maxv) maxv = value;
    }
    long long range = (long long)maxv - (long long)minv + 1LL;
    return range > 0 && range <= 2000000LL;
}

static void writeLearningRow(const MemoryEntry &entry, bool appendMode) {
    ios::openmode mode = ios::out;
    if (appendMode) mode |= ios::app;
    ofstream file(LEARNING_FILE, mode);
    if (!file.is_open()) return;

    file << setprecision(17);

    for (int i = 0; i < FEATURE_COUNT; ++i) {
        if (i) file << ',';
        file << entry.features[i];
    }
    file << ',' << entry.algo;
    file << '\n';
}

static void writeNormalizationParams(const array<double, FEATURE_COUNT> &mean,
                                     const array<double, FEATURE_COUNT> &stdev) {
    ofstream file(NORMALIZATION_FILE, ios::out | ios::trunc);
    if (!file.is_open()) return;

    file << setprecision(17);

    for (int i = 0; i < FEATURE_COUNT; ++i) {
        if (i) file << ',';
        file << mean[i];
    }
    for (int i = 0; i < FEATURE_COUNT; ++i) {
        file << ',' << stdev[i];
    }
    file << '\n';
}

static void resetLearningFile() {
    ofstream file(LEARNING_FILE, ios::out | ios::trunc);
}

static bool readNormalizationParams(array<double, FEATURE_COUNT> &mean,
                                    array<double, FEATURE_COUNT> &stdev) {
    mean.fill(0.0);
    stdev.fill(1.0);
    ifstream file(NORMALIZATION_FILE);
    if (!file.is_open()) return false;

    string line;
    if (!getline(file, line) || line.empty()) return false;
    vector<string> fields;
    string current;
    stringstream ss(line);
    while (getline(ss, current, ',')) {
        fields.push_back(current);
    }
    if (fields.size() != FEATURE_COUNT * 2) return false;

    for (int i = 0; i < FEATURE_COUNT; ++i) {
        mean[i] = stod(fields[i]);
    }
    for (int i = 0; i < FEATURE_COUNT; ++i) {
        stdev[i] = stod(fields[FEATURE_COUNT + i]);
    }
    return true;
}

static vector<string> splitCsvLine(const string &line) {
    vector<string> fields;
    string current;
    stringstream ss(line);
    while (getline(ss, current, ',')) {
        fields.push_back(current);
    }
    return fields;
}

static vector<MemoryEntry> readFromFile() {
    vector<MemoryEntry> data;
    ifstream file(LEARNING_FILE);
    if (!file.is_open()) return data;

    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        vector<string> fields = splitCsvLine(line);
        if (fields.size() == FEATURE_COUNT + 1) {
            MemoryEntry entry;
            for (int i = 0; i < FEATURE_COUNT; ++i) entry.features[i] = stod(fields[i]);
            entry.algo = fields[FEATURE_COUNT];
            entry.size = (int)entry.features[0];
            entry.times.fill(-1.0);
            data.push_back(entry);
        }
    }
    return data;
}

static void rebuildModel(const vector<MemoryEntry> &mem);
static void sortByAlgorithm(vector<int> &arr, const string &algo);

static bool loadTrainedModel(vector<MemoryEntry> &mem) {
    if (!readNormalizationParams(gFeatureMean, gFeatureStd)) {
        return false;
    }

    mem = readFromFile();
    if (mem.empty()) {
        return false;
    }

    rebuildModel(mem);
    return gModelReady;
}

static vector<vector<int>> readArrayFile(const string &filename) {
    vector<vector<int>> arrays;
    ifstream file(filename);
    if (!file.is_open()) return arrays;

    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        vector<int> arr;
        int value;
        while (ss >> value) {
            arr.push_back(value);
        }
        if (!arr.empty()) arrays.push_back(arr);
    }
    return arrays;
}

static vector<int> generateOneRandomArray(mt19937_64 &rng) {
    uniform_int_distribution<int> lengthDist(MIN_GENERATED_LENGTH, MAX_GENERATED_LENGTH);
    uniform_int_distribution<int> typeDist(0, 11);
    uniform_int_distribution<long long> valueDist(1LL, MAX_GENERATED_VALUE);
    uniform_real_distribution<double> probability(0.0, 1.0);

    int n = lengthDist(rng);
    int kind = typeDist(rng);

    vector<int> arr(n);
    auto randomValue = [&]() -> int {
        return (int)valueDist(rng);
    };

    auto randomDistinctPool = [&](int distinct) {
        vector<int> pool(distinct);
        for (int i = 0; i < distinct; ++i) pool[i] = randomValue();
        return pool;
    };

    switch (kind) {
    case 0: {
        for (int i = 0; i < n; ++i) arr[i] = randomValue();
        break;
    }
    case 1: {
        for (int i = 0; i < n; ++i) arr[i] = randomValue();
        sort(arr.begin(), arr.end());
        break;
    }
    case 2: {
        for (int i = 0; i < n; ++i) arr[i] = randomValue();
        sort(arr.begin(), arr.end(), greater<int>());
        break;
    }
    case 3: {
        int value = randomValue();
        fill(arr.begin(), arr.end(), value);
        break;
    }
    case 4: {
        int distinct = uniform_int_distribution<int>(2, min(6, n))(rng);
        vector<int> pool = randomDistinctPool(distinct);
        for (int i = 0; i < n; ++i) arr[i] = pool[uniform_int_distribution<int>(0, distinct - 1)(rng)];
        break;
    }
    case 5: {
        int a = randomValue();
        int b = randomValue();
        for (int i = 0; i < n; ++i) arr[i] = (i % 2 == 0) ? a : b;
        break;
    }
    case 6: {
        int blockCount = uniform_int_distribution<int>(2, min(12, n))(rng);
        vector<int> pool = randomDistinctPool(blockCount);
        int blockSize = max(1, n / blockCount);
        for (int i = 0; i < n; ++i) {
            arr[i] = pool[min(blockCount - 1, i / blockSize)];
        }
        break;
    }
    case 7: {
        for (int i = 0; i < n; ++i) arr[i] = randomValue();
        sort(arr.begin(), arr.end());
        int swaps = max(1, n / 25);
        for (int i = 0; i < swaps; ++i) {
            int left = uniform_int_distribution<int>(0, n - 1)(rng);
            int right = uniform_int_distribution<int>(0, n - 1)(rng);
            swap(arr[left], arr[right]);
        }
        break;
    }
    case 8: {
        for (int i = 0; i < n; ++i) arr[i] = randomValue();
        sort(arr.begin(), arr.end(), greater<int>());
        int swaps = max(1, n / 25);
        for (int i = 0; i < swaps; ++i) {
            int left = uniform_int_distribution<int>(0, n - 1)(rng);
            int right = uniform_int_distribution<int>(0, n - 1)(rng);
            swap(arr[left], arr[right]);
        }
        break;
    }
    case 9: {
        long long low = valueDist(rng);
        long long high = min<long long>(MAX_GENERATED_VALUE, low + uniform_int_distribution<int>(3, 5000)(rng));
        if (high < low) high = low;
        uniform_int_distribution<long long> narrow(low, high);
        for (int i = 0; i < n; ++i) arr[i] = (int)narrow(rng);
        break;
    }
    case 10: {
        int runValue = randomValue();
        int remaining = n;
        int pos = 0;
        while (remaining > 0) {
            int runLength = min(remaining, uniform_int_distribution<int>(1, max(1, n / 10))(rng));
            if (probability(rng) < 0.35) {
                runValue = randomValue();
            }
            for (int i = 0; i < runLength; ++i) arr[pos + i] = runValue;
            pos += runLength;
            remaining -= runLength;
        }
        break;
    }
    case 11: {
        int half = max(1, n / 2);
        for (int i = 0; i < half; ++i) arr[i] = randomValue();
        int repeated = randomValue();
        for (int i = half; i < n; ++i) arr[i] = (probability(rng) < 0.8) ? repeated : randomValue();
        if (probability(rng) < 0.5) {
            shuffle(arr.begin(), arr.end(), rng);
        }
        break;
    }
    }

    if (probability(rng) < 0.2) {
        int swapCount = max(1, n / 30);
        for (int i = 0; i < swapCount; ++i) {
            int left = uniform_int_distribution<int>(0, n - 1)(rng);
            int right = uniform_int_distribution<int>(0, n - 1)(rng);
            swap(arr[left], arr[right]);
        }
    }

    return arr;
}

static bool generateRandomArraysFile(const string &filename, int count) {
    if (count < MIN_GENERATED_ARRAYS) count = MIN_GENERATED_ARRAYS;
    ofstream file(filename, ios::out | ios::trunc);
    if (!file.is_open()) return false;

    random_device rd;
    mt19937_64 rng(rd());
    for (int i = 0; i < count; ++i) {
        vector<int> arr = generateOneRandomArray(rng);
        for (size_t j = 0; j < arr.size(); ++j) {
            if (j) file << ' ';
            file << arr[j];
        }
        file << "\n\n";
        if ((i + 1) % 1000 == 0) {
            cout << "Generated " << (i + 1) << " arrays...\n";
        }
    }
    return true;
}

static string trainOnArray(const vector<int> &baseArr, array<double, ALGO_COUNT> &times) {
    times.fill(-1.0);
    if (baseArr.empty()) return "NONE";

    string bestAlgo = "NONE";
    double bestTime = numeric_limits<double>::infinity();

    auto medianTime = [](vector<double> &samples) -> double {
        if (samples.empty()) return numeric_limits<double>::infinity();
        sort(samples.begin(), samples.end());
        size_t mid = samples.size() / 2;
        if (samples.size() % 2 == 0) {
            return 0.5 * (samples[mid - 1] + samples[mid]);
        }
        return samples[mid];
    };

    for (int i = 0; i < ALGO_COUNT; ++i) {
        vector<int> copy = baseArr;
        const string &algo = ALGO_NAMES[i];

        if (algo == "CountingSort" && !countingSortAllowed(copy)) continue;
        if (algo == "RadixSort" && !allNonNegative(copy)) continue;

        vector<double> runs;
        runs.reserve(TRAIN_BENCH_REPEATS);
        for (int repeat = 0; repeat < TRAIN_BENCH_REPEATS; ++repeat) {
            copy = baseArr;
            auto start = high_resolution_clock::now();
            if (algo == "InsertionSort") {
                insertionSort(copy);
            } else if (algo == "CountingSort") {
                countingSort(copy);
            } else if (algo == "RadixSort") {
                radixSort(copy);
            } else if (algo == "QuickSort") {
                quickSort(copy, 0, (int)copy.size() - 1);
            } else if (algo == "MergeSort") {
                mergeSort(copy, 0, (int)copy.size() - 1);
            }
            auto end = high_resolution_clock::now();
            runs.push_back(duration<double, milli>(end - start).count());
        }
        double elapsed = medianTime(runs);
        times[i] = elapsed;
        if (elapsed < bestTime) {
            bestTime = elapsed;
            bestAlgo = algo;
        }
    }

    return bestAlgo;
}

static int buildKDTree(vector<KDNode> &nodes, vector<int> &indices, const vector<array<double, FEATURE_COUNT>> &points, int depth) {
    if (indices.empty()) return -1;

    int dim = depth % FEATURE_COUNT;
    size_t mid = indices.size() / 2;
    nth_element(indices.begin(), indices.begin() + (ptrdiff_t)mid, indices.end(), [&](int a, int b) {
        return points[a][dim] < points[b][dim];
    });

    int nodeIndex = (int)nodes.size();
    nodes.push_back({});
    int pivotIndex = indices[mid];
    nodes[nodeIndex].index = pivotIndex;
    nodes[nodeIndex].dim = dim;
    nodes[nodeIndex].split = points[pivotIndex][dim];

    vector<int> left(indices.begin(), indices.begin() + (ptrdiff_t)mid);
    vector<int> right(indices.begin() + (ptrdiff_t)mid + 1, indices.end());
    nodes[nodeIndex].left = buildKDTree(nodes, left, points, depth + 1);
    nodes[nodeIndex].right = buildKDTree(nodes, right, points, depth + 1);
    return nodeIndex;
}

static void computeNormalization(const vector<MemoryEntry> &mem,
                                 array<double, FEATURE_COUNT> &mean,
                                 array<double, FEATURE_COUNT> &stdev) {
    mean.fill(0.0);
    stdev.fill(0.0);
    if (mem.empty()) return;

    for (const auto &entry : mem) {
        for (int i = 0; i < FEATURE_COUNT; ++i) {
            mean[i] += entry.features[i];
        }
    }
    for (int i = 0; i < FEATURE_COUNT; ++i) {
        mean[i] /= (double)mem.size();
    }

    for (const auto &entry : mem) {
        for (int i = 0; i < FEATURE_COUNT; ++i) {
            double diff = entry.features[i] - mean[i];
            stdev[i] += diff * diff;
        }
    }

    for (int i = 0; i < FEATURE_COUNT; ++i) {
        stdev[i] = sqrt(stdev[i] / (double)mem.size());
        if (stdev[i] < 1e-9) stdev[i] = 1.0;
    }
}

static array<double, FEATURE_COUNT> normalizeFeatures(const array<double, FEATURE_COUNT> &features,
                                                      const array<double, FEATURE_COUNT> &mean,
                                                      const array<double, FEATURE_COUNT> &stdev) {
    array<double, FEATURE_COUNT> normalized{};
    for (int i = 0; i < FEATURE_COUNT; ++i) {
        normalized[i] = (features[i] - mean[i]) / stdev[i];
    }
    return normalized;
}

static void rebuildModel(const vector<MemoryEntry> &mem) {
    gMemory = mem;
    gNormalizedPoints.clear();
    gKDNodes.clear();
    gWinningAlgoIndex.clear();
    gKDRoot = -1;
    gModelReady = false;

    if (gMemory.empty()) return;

    gNormalizedPoints.resize(gMemory.size());
    gWinningAlgoIndex.resize(gMemory.size(), -1);

    for (size_t i = 0; i < gMemory.size(); ++i) {
        gNormalizedPoints[i] = gMemory[i].features;
        gWinningAlgoIndex[i] = algoIndexFromName(gMemory[i].algo);
    }

    vector<int> indices(gMemory.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = (int)i;
    gKDRoot = buildKDTree(gKDNodes, indices, gNormalizedPoints, 0);
    gModelReady = (gKDRoot != -1);
}

static void kdSearch(const vector<KDNode> &nodes,
                     int nodeIndex,
                     const vector<array<double, FEATURE_COUNT>> &points,
                     const array<double, FEATURE_COUNT> &target,
                     int k,
                     priority_queue<pair<double, int>> &best) {
    if (nodeIndex == -1) return;

    const KDNode &node = nodes[nodeIndex];
    int pointIndex = node.index;
    double distSq = 0.0;
    for (int i = 0; i < FEATURE_COUNT; ++i) {
        double diff = target[i] - points[pointIndex][i];
        distSq += diff * diff;
    }
    double dist = sqrt(distSq);
    if ((int)best.size() < k) {
        best.push({dist, pointIndex});
    } else if (dist < best.top().first) {
        best.pop();
        best.push({dist, pointIndex});
    }

    int dim = node.dim;
    double targetValue = target[dim];
    int first = targetValue < node.split ? node.left : node.right;
    int second = targetValue < node.split ? node.right : node.left;

    if (first != -1) kdSearch(nodes, first, points, target, k, best);

    double axisDiff = targetValue - node.split;
    double worst = best.empty() ? numeric_limits<double>::infinity() : best.top().first;
    if (second != -1 && ((int)best.size() < k || axisDiff * axisDiff <= worst * worst)) {
        kdSearch(nodes, second, points, target, k, best);
    }
}

static bool isAlgoApplicable(const string &algo, const vector<int> &arr) {
    if (algo == "CountingSort") return countingSortAllowed(arr);
    if (algo == "RadixSort") return allNonNegative(arr);
    return !arr.empty();
}

static double benchmarkSingleRun(const vector<int> &arr, const string &algo) {
    if (arr.empty() || algo == "NONE") return numeric_limits<double>::infinity();
    if (!isAlgoApplicable(algo, arr)) return numeric_limits<double>::infinity();

    vector<int> copy = arr;
    auto start = high_resolution_clock::now();
    sortByAlgorithm(copy, algo);
    auto end = high_resolution_clock::now();
    return duration<double, milli>(end - start).count();
}

static string predictWithKDTree(const array<double, FEATURE_COUNT> &features,
                                const vector<int> *sourceArray) {
    if (!gModelReady || gMemory.empty()) return "NONE";

    array<double, FEATURE_COUNT> target = normalizeFeatures(features, gFeatureMean, gFeatureStd);
    priority_queue<pair<double, int>> best;
    int k = min(KD_NEIGHBOR_COUNT, (int)gMemory.size());
    kdSearch(gKDNodes, gKDRoot, gNormalizedPoints, target, k, best);

    unordered_map<string, double> score;
    while (!best.empty()) {
        pair<double, int> top = best.top();
        best.pop();
        int index = top.second;
        const string &algo = gMemory[index].algo;
        if (algo == "NONE") continue;
        score[algo] += 1.0 / (top.first + VOTING_EPS);
    }

    if (sourceArray != nullptr) {
        vector<string> toRemove;
        toRemove.reserve(score.size());
        for (const auto &entry : score) {
            if (!isAlgoApplicable(entry.first, *sourceArray)) {
                toRemove.push_back(entry.first);
            }
        }
        for (const string &algo : toRemove) {
            score.erase(algo);
        }
    }

    if (score.empty()) return "NONE";

    string bestAlgo = "NONE";
    double bestScore = -1.0;
    for (const auto &entry : score) {
        if (entry.second > bestScore) {
            bestScore = entry.second;
            bestAlgo = entry.first;
        }
    }

    if (sourceArray != nullptr && score.size() >= 2) {
        vector<pair<string, double>> ranked(score.begin(), score.end());
        sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b) {
            return a.second > b.second;
        });

        const string &firstAlgo = ranked[0].first;
        const string &secondAlgo = ranked[1].first;
        double firstScore = ranked[0].second;
        double secondScore = ranked[1].second;

        if (firstScore > 0.0 && (secondScore / firstScore) >= CLOSE_SCORE_RATIO) {
            double firstTime = benchmarkSingleRun(*sourceArray, firstAlgo);
            double secondTime = benchmarkSingleRun(*sourceArray, secondAlgo);
            if (secondTime < firstTime) {
                bestAlgo = secondAlgo;
            } else {
                bestAlgo = firstAlgo;
            }
        }
    }

    return bestAlgo;
}

static void sortByAlgorithm(vector<int> &arr, const string &algo) {
    if (arr.empty() || algo == "NONE") return;
    if (algo == "InsertionSort") {
        insertionSort(arr);
    } else if (algo == "CountingSort") {
        countingSort(arr);
    } else if (algo == "RadixSort") {
        radixSort(arr);
    } else if (algo == "QuickSort") {
        quickSort(arr, 0, (int)arr.size() - 1);
    } else if (algo == "MergeSort") {
        mergeSort(arr, 0, (int)arr.size() - 1);
    }
}

static void showMemory(const vector<MemoryEntry> &mem) {
    if (mem.empty()) {
        cout << "No stored memory found.\n";
        return;
    }

    cout << "features[0..4], algo\n";
    for (const auto &entry : mem) {
        for (int i = 0; i < FEATURE_COUNT; ++i) {
            cout << entry.features[i] << (i + 1 < FEATURE_COUNT ? ", " : "");
        }
        cout << " | algo=" << entry.algo << '\n';
    }
}

static void printArray(const vector<int> &arr) {
    for (size_t i = 0; i < arr.size(); ++i) {
        if (i) cout << ' ';
        cout << arr[i];
    }
    cout << '\n';
}

static void printFeatureSummary(const array<double, FEATURE_COUNT> &features) {
    cout << "  Features:\n";
    cout << left << setw(28) << "    Size" << ": " << features[0] << '\n';
    cout << left << setw(28) << "    Non-decreasing ratio" << ": " << features[1] << '\n';
    cout << left << setw(28) << "    Distinct ratio" << ": " << features[2] << '\n';
    cout << left << setw(28) << "    Range/N" << ": " << features[3] << '\n';
    cout << left << setw(28) << "    Entropy" << ": " << features[4] << '\n';
}

static void runBruteForceBenchmarkOnTestingFile() {
    vector<vector<int>> testArrays = readArrayFile("testing.prn");
    cout << "Found " << testArrays.size() << " test arrays.\n";
    if (testArrays.empty()) {
        cout << "No arrays to test.\n";
        return;
    }

    for (size_t i = 0; i < testArrays.size(); ++i) {
        const vector<int> &source = testArrays[i];
        if (source.empty()) continue;

        cout << "\n=================================\n";
        cout << "Testing array " << (i + 1) << " / " << testArrays.size() << '\n';
        cout << "=================================\n";

        array<double, FEATURE_COUNT> features = extractFeatures(source);
        printFeatureSummary(features);

        string bestAlgo = "NONE";
        double bestTime = numeric_limits<double>::infinity();

        for (int algoIndex = 0; algoIndex < ALGO_COUNT; ++algoIndex) {
            vector<int> copy = source;
            const string &algo = ALGO_NAMES[algoIndex];

            cout << "  Running " << algo << "... " << flush;
            auto start = high_resolution_clock::now();
            if (algo == "InsertionSort") {
                insertionSort(copy);
            } else if (algo == "CountingSort") {
                countingSort(copy);
            } else if (algo == "RadixSort") {
                radixSort(copy);
            } else if (algo == "QuickSort") {
                quickSort(copy, 0, (int)copy.size() - 1);
            } else if (algo == "MergeSort") {
                mergeSort(copy, 0, (int)copy.size() - 1);
            }
            auto end = high_resolution_clock::now();
            double elapsed = duration<double, milli>(end - start).count();

            cout << elapsed << " ms\n";
            if (elapsed < bestTime) {
                bestTime = elapsed;
                bestAlgo = algo;
            }
        }

        cout << "  Fastest algo: " << bestAlgo << " (" << bestTime << " ms)\n";
    }
}

#ifdef BRUTE_FORCE_SORTER
int main() {
    cout << "=================================\n";
    cout << "Sorter (Brute Force Benchmark)\n";
    cout << "=================================\n";
    runBruteForceBenchmarkOnTestingFile();
    return 0;
}
#else
int main() {
    vector<MemoryEntry> mem = readFromFile();
    if (!loadTrainedModel(mem)) {
        mem.clear();
    }

    while (true) {
        int choice = 0;
        cout << "\n=================================\n";
        cout << "Hybrid Trained Sorter (KD-tree)\n";
        cout << "=================================\n";
        cout << "1. Train from array file\n";
        cout << "2. Test and sort array file\n";
        cout << "3. Show stored memory\n";
        cout << "4. Exit\n";
        cout << "Enter Choice: ";
        cin >> choice;

        if (choice == 4) {
            break;
        }

        if (choice == 1) {
            vector<vector<int>> arrays = readArrayFile("training.prn");
            cout << "Found " << arrays.size() << " training arrays.\n";
            if (arrays.empty()) {
                cout << "No arrays to train on.\n";
                continue;
            }

            cout << "Generating learning.csv from training.prn...\n";
            resetLearningFile();
            vector<MemoryEntry> rawMem;
            for (size_t i = 0; i < arrays.size(); ++i) {
                const vector<int> &arr = arrays[i];
                MemoryEntry entry;
                entry.features = extractFeatures(arr);
                entry.size = (int)arr.size();
                entry.algo = trainOnArray(arr, entry.times);
                rawMem.push_back(entry);
                if ((i + 1) % 100 == 0 || i + 1 == arrays.size()) {
                    cout << "Processed " << (i + 1) << " / " << arrays.size() << " arrays\n";
                }
            }

            computeNormalization(rawMem, gFeatureMean, gFeatureStd);
            vector<MemoryEntry> normalizedMem = rawMem;
            for (auto &entry : normalizedMem) {
                entry.features = normalizeFeatures(entry.features, gFeatureMean, gFeatureStd);
            }

            writeNormalizationParams(gFeatureMean, gFeatureStd);
            resetLearningFile();
            for (size_t i = 0; i < normalizedMem.size(); ++i) {
                writeLearningRow(normalizedMem[i], i != 0);
                if ((i + 1) % 100 == 0 || i + 1 == normalizedMem.size()) {
                    cout << "Normalized write " << (i + 1) << " / " << normalizedMem.size() << " rows\n";
                }
            }

            cout << "Reloading learning.csv and rebuilding KD-tree...\n";
            if (!loadTrainedModel(mem)) {
                cout << "Failed to load learning artifacts after training.\n";
                continue;
            }
            cout << "Training complete. learning.csv has " << mem.size() << " rows and the KD-tree is ready.\n";
        } else if (choice == 2) {
            if (!gModelReady && !loadTrainedModel(mem)) {
                cout << "Cannot run option 2: learning.csv and/or learning_norm.csv is missing or invalid. Train first.\n";
                continue;
            }

            vector<vector<int>> testArrays = readArrayFile("testing.prn");
            cout << "Found " << testArrays.size() << " test arrays.\n";
            if (testArrays.empty()) {
                cout << "No arrays to test.\n";
                continue;
            }

            for (size_t i = 0; i < testArrays.size(); ++i) {
                vector<int> arr = testArrays[i];
                if (arr.empty()) continue;

                array<double, FEATURE_COUNT> features = extractFeatures(arr);

                string algo = predictWithKDTree(features, &arr);
                cout << "Array " << (i + 1) << " predicted algo: " << algo << '\n';
            }
        } else if (choice == 3) {
            if (mem.empty()) mem = readFromFile();
            showMemory(mem);
        } else {
            cout << "Invalid Choice\n";
        }
    }

    return 0;
}
#endif

