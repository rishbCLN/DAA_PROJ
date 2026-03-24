#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace std;

static const int MIN_GENERATED_ARRAYS = 10100;
static const int TEST_GENERATED_ARRAYS = 2;
static const int MIN_GENERATED_LENGTH = 7000;
static const int MAX_GENERATED_LENGTH = 100000;
static const long long MAX_GENERATED_VALUE = 1000000LL;

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

    auto makePool = [&](int distinct) {
        vector<int> pool(distinct);
        for (int i = 0; i < distinct; ++i) {
            pool[i] = randomValue();
        }
        return pool;
    };

    switch (kind) {
    case 0:
        for (int i = 0; i < n; ++i) arr[i] = randomValue();
        break;
    case 1:
        for (int i = 0; i < n; ++i) arr[i] = randomValue();
        sort(arr.begin(), arr.end());
        break;
    case 2:
        for (int i = 0; i < n; ++i) arr[i] = randomValue();
        sort(arr.begin(), arr.end(), greater<int>());
        break;
    case 3: {
        int value = randomValue();
        fill(arr.begin(), arr.end(), value);
        break;
    }
    case 4: {
        int distinct = uniform_int_distribution<int>(2, min(6, n))(rng);
        vector<int> pool = makePool(distinct);
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
        vector<int> pool = makePool(blockCount);
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
            if (probability(rng) < 0.35) runValue = randomValue();
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

static bool generateRandomArraysFile(const string &filename, int count, int minimumCount = MIN_GENERATED_ARRAYS, bool blankLineBetweenArrays = true) {
    if (count < minimumCount) count = minimumCount;

    ofstream file(filename, ios::out | ios::trunc);
    if (!file.is_open()) return false;

    random_device rd;
    mt19937_64 rng(rd());

    cout << "Generating " << count << " arrays for " << filename << "...\n";

    for (int i = 0; i < count; ++i) {
        vector<int> arr = generateOneRandomArray(rng);
        for (size_t j = 0; j < arr.size(); ++j) {
            if (j) file << ' ';
            file << arr[j];
        }
        file << (blankLineBetweenArrays ? "\n\n" : "\n");

        if ((i + 1) % 1000 == 0 || i + 1 == count) {
            cout << "\r  Progress: " << (i + 1) << "/" << count << " arrays" << flush;
        }
    }

    cout << "\r  Progress: " << count << "/" << count << " arrays\n";

    return true;
}

int main() {
    while (true) {
        int choice = 0;
        cout << "\n=================================\n";
        cout << "Random Array Generator\n";
        cout << "=================================\n";
        cout << "1. Generate training.prn\n";
        cout << "2. Generate testing.prn\n";
        cout << "3. Generate custom file\n";
        cout << "4. Exit\n";
        cout << "Enter Choice: ";
        cin >> choice;

        if (choice == 4) break;

        if (choice == 1) {
            if (generateRandomArraysFile("training.prn", MIN_GENERATED_ARRAYS)) {
                cout << "Generated " << MIN_GENERATED_ARRAYS << " arrays in training.prn\n";
            } else {
                cout << "Failed to generate training.prn\n";
            }
        } else if (choice == 2) {
            if (generateRandomArraysFile("testing.prn", 1, 1, true)) {
                cout << "Generated 1 array in testing.prn\n";
            } else {
                cout << "Failed to generate testing.prn\n";
            }
        } else if (choice == 3) {
            string outputFile;
            int count = MIN_GENERATED_ARRAYS;
            cout << "Enter output filename: ";
            cin >> outputFile;
            cout << "Enter number of arrays to generate (minimum 10100): ";
            cin >> count;
            if (count < MIN_GENERATED_ARRAYS) count = MIN_GENERATED_ARRAYS;
            if (generateRandomArraysFile(outputFile, count)) {
                cout << "Generated " << count << " arrays in " << outputFile << "\n";
            } else {
                cout << "Failed to generate " << outputFile << "\n";
            }
        } else {
            cout << "Invalid choice\n";
        }
    }

    return 0;
}