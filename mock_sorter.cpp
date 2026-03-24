#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;
using namespace std::chrono;

static const int FEATURE_COUNT = 5;
static const int ALGO_COUNT = 5;
static const array<string, ALGO_COUNT> ALGO_NAMES = {
    "InsertionSort",
    "CountingSort",
    "RadixSort",
    "QuickSort",
    "MergeSort"
};

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
        if (i > 0 && a[i] >= a[i - 1]) {
            ++nondecreasing;
        }
    }

    features[0] = (double)n;
    features[1] = (n > 1) ? (double)nondecreasing / (double)(n - 1) : 1.0;
    features[2] = (double)counts.size() / (double)n;
    features[3] = (double)(maxv - minv) / (double)max(1, n);
    features[4] = calcEntropy(a);
    return features;
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

static void printFeatureSummary(const array<double, FEATURE_COUNT> &features) {
    cout << "  Features:\n";
    cout << left << setw(28) << "    Size" << ": " << features[0] << '\n';
    cout << left << setw(28) << "    Non-decreasing ratio" << ": " << features[1] << '\n';
    cout << left << setw(28) << "    Distinct ratio" << ": " << features[2] << '\n';
    cout << left << setw(28) << "    Range/N" << ": " << features[3] << '\n';
    cout << left << setw(28) << "    Entropy" << ": " << features[4] << '\n';
}

static void printArray(const vector<int> &arr) {
    for (size_t i = 0; i < arr.size(); ++i) {
        if (i) cout << ' ';
        cout << arr[i];
    }
    cout << '\n';
}

int main() {
    vector<vector<int>> testArrays = readArrayFile("testing.prn");
    cout << "=================================\n";
    cout << "mock_sorter (Brute Force Runner)\n";
    cout << "=================================\n";
    cout << "Found " << testArrays.size() << " test arrays in testing.prn.\n";

    if (testArrays.empty()) {
        cout << "No arrays to test.\n";
        return 0;
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

            cout << fixed << setprecision(3) << elapsed << " ms\n";
            if (elapsed < bestTime) {
                bestTime = elapsed;
                bestAlgo = algo;
            }
        }

        cout << "  Winner: " << bestAlgo << " (" << fixed << setprecision(3) << bestTime << " ms)\n";
    }

    cout << "\nDone. Press Enter to close...";
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
    cin.get();

    return 0;
}
