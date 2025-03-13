#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <queue>

using namespace std;

// Image dimensions
const int ROWS = 420;
const int COLS = 560;

// Constants
const double M_PI = 3.1415926535897932384626433832; //PI constant
const double deltaTheta = 0.1;
const int h = static_cast<int>(sqrt(2)  * sqrt(ROWS * ROWS + COLS * COLS) / 2.0);
const int maxRho = h * 2; //total range of Rho
const int numTheta = 180; // total range of theta in degrees

// Input and result images
unsigned char in_buf[ROWS][COLS];
unsigned char laplacian_result[ROWS][COLS];
unsigned char sobel_result[ROWS][COLS];
unsigned char result[ROWS][COLS];
unsigned char hough_result[ROWS][COLS]; 

// Accumulator vector for the hough transform
vector<vector<unsigned int>> A(maxRho, vector<unsigned int>(numTheta,0));
// pair array to hold the top 3 lines, their lengh , Rho and theta 
pair<unsigned int, pair<int, int>> topLines[3] = { {0, {0, 0}}, {0, {0, 0}}, {0, {0, 0}} };

// Threshold values
const int sobel_threshold = 128;
const int hough_threshold = 64;
const int result_threshold = 48;

// Function prototypes
void sobelOperator(unsigned char in_img[][COLS], unsigned char out_img[][COLS]);
void laplacianOperator(unsigned char in_img[][COLS], unsigned char out_img[][COLS]);
void houghTransform(unsigned char edge_img[][COLS], unsigned char result[][COLS]);

int main() {
    // File handling
    ifstream fin;
    ofstream foutLaplacian, foutSobel, foutCombined, foutHough;

    // Open input image file
    fin.open("building560x520.raw", ios::binary);
    if (!fin.is_open()) {
        cerr << "ERROR: Cannot open input image file" << endl;
        exit(1);
    }

    foutLaplacian.open("laplacian_result.raw", ios::binary);
    if (!foutLaplacian.is_open()) {
        cerr << "ERROR: Cannot open output image file for Laplacian result" << endl;
        exit(1);
    }

    foutSobel.open("sobel_result.raw", ios::binary);
    if (!foutSobel.is_open()) {
        cerr << "ERROR: Cannot open output image file for Sobel result" << endl;
        exit(1);
    }

    foutCombined.open("combined_result.raw", ios::binary);
    if (!foutCombined.is_open()) {
        cerr << "ERROR: Cannot open output image file for Combined result" << endl;
        exit(1);
    }

    foutHough.open("hough_result.raw", ios::binary);
    if (!foutHough.is_open()) {
        cerr << "ERROR: Cannot open output image file for Hough result" << endl;
        exit(1);
    }

    // Load input image
    cout << "... Load input image" << endl;
    fin.read(reinterpret_cast<char*>(in_buf), ROWS * COLS);
    if (!fin) {
        cerr << "ERROR: Read input image file error" << endl;
        exit(1);
    }

    // Apply Sobel operator
    sobelOperator(in_buf, sobel_result);
   

   // Apply Laplacian operator
    laplacianOperator(in_buf, laplacian_result);

    // Combine the results
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            result[i][j] = max(sobel_result[i][j], laplacian_result[i][j]);
        }
    }

    // Threshold the combined result
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            result[i][j] = (result[i][j] > result_threshold) ? 255 : 0;
        }
    }

    // Perform Hough transform
    houghTransform(result, hough_result);

    // Save Laplacian result
    cout << "... Save Laplacian result" << endl;
    foutLaplacian.write(reinterpret_cast<const char*>(laplacian_result), ROWS * COLS * sizeof(unsigned char));
    if (!foutLaplacian) {
        cerr << "ERROR: Write output image file error for Laplacian result" << endl;
        exit(1);
    }

    // Save Sobel result
    cout << "... Save Sobel result" << endl;
    foutSobel.write(reinterpret_cast<const char*>(sobel_result), ROWS * COLS * sizeof(unsigned char));
    if (!foutSobel) {
        cerr << "ERROR: Write output image file error for Sobel result" << endl;
        exit(1);
    }

    // Save Combined result
    cout << "... Save Combined result" << endl;
    foutCombined.write(reinterpret_cast<const char*>(result), ROWS * COLS * sizeof(unsigned char));
    if (!foutCombined) {
        cerr << "ERROR: Write output image file error for Combined result" << endl;
        exit(1);
    }

    // Save Hough result
    cout << "... Save Hough result" << endl;
    foutHough.write(reinterpret_cast<const char*>(hough_result), ROWS * COLS * sizeof(unsigned char));
    if (!foutHough) {
        cerr << "ERROR: Write output image file error for Hough result" << endl;
        exit(1);
    }

    // Close file streams
    fin.close();
    foutLaplacian.close();
    foutSobel.close();
    foutCombined.close();
    foutHough.close();
    return 0;
}

// Sobel Operator Function
void sobelOperator(unsigned char in_img[][COLS], unsigned char out_img[][COLS]) {

    // Sobel kernels
    int sobelX[3][3] = { {-1, 0, 1},
                        {-2, 0, 2},
                        {-1, 0, 1} };

    int sobelY[3][3] = { {-1, -2, -1},
                        {0, 0, 0},
                        {1, 2, 1} };

    // Apply Sobel operator to each pixel
    for (int i = 1; i < ROWS - 1; ++i) {
        for (int j = 1; j < COLS - 1; ++j) {
            int gradientX = 0;
            int gradientY = 0;

            // Convolution with Sobel kernels
            for (int m = 0; m < 3; ++m) {
                for (int n = 0; n < 3; ++n) {
                    gradientX += sobelX[m][n] * in_img[i + m - 1][j + n - 1];
                    gradientY += sobelY[m][n] * in_img[i + m - 1][j + n - 1];
                }
            }

            // Compute magnitude of the gradient
            int magnitude = sqrt(gradientX * gradientX + gradientY * gradientY);

            // Clamp magnitude to be within [0, 255]
            out_img[i][j] = static_cast<unsigned char>(min(255, max(0, magnitude)));
        }
    }

    // Threshold the Sobel result
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            out_img[i][j] = (out_img[i][j] > sobel_threshold) ? 255 : 0;
        }
    }

}

// Laplacian Operator Function
void laplacianOperator(unsigned char in_img[][COLS], unsigned char out_img[][COLS]) {
    // Laplacian operator kernel
    int laplacianKernel[3][3] = {
        {0, 1, 0},
        {1, -4, 1},
        {0, 1, 0}
    };

    // Apply Laplacian operator to each pixel
    for (int i = 1; i < ROWS - 1; ++i) {
        for (int j = 1; j < COLS - 1; ++j) {
            int laplacianSum = 0;

            // Convolution with Laplacian kernel
            for (int m = 0; m < 3; ++m) {
                for (int n = 0; n < 3; ++n) {
                    laplacianSum += laplacianKernel[m][n] * in_img[i + m - 1][j + n - 1];
                }
            }

            // Clamp the result to be within [0, 255]
            out_img[i][j] = static_cast<unsigned char>(min(255, max(0, laplacianSum)));
        }
    }

}

// Hough Transform Function
void houghTransform(unsigned char edge_img[][COLS], unsigned char result[][COLS]) {

    // Perform Hough Transform
    for (int xi = 0; xi < ROWS; xi++) {
        for (int yi = 0; yi < COLS; yi++) {
            if (edge_img[xi][yi] == 255 ) { // Edge point found and gradient is greater than threshold
                for (int theta = 0; theta < numTheta; theta++) {
                    double rho = xi * cos(theta * M_PI / 180) - yi * sin(theta * M_PI / 180);
                    
                    if ((int)(round(rho + h)) < A.size() ) {
                        A[(int)(round(rho + h))][theta]++;
                    }
                }
            }
        }
    }


    vector<pair<unsigned int, pair<int, int>>> potentialLines;

    // Find peaks and draw lines
    for (int r = 0; r < maxRho; ++r) {
        for (int t = 0; t < numTheta; ++t) {
            if (r > 0 && t > 0 && r < A.size() - 1 && t < A[0].size() - 1 &&
                A[r][t] > hough_threshold &&
                A[r][t] > A[r - 1][t] &&
                A[r][t] > A[r + 1][t] &&
                A[r][t] > A[r][t - 1] &&
                A[r][t] > A[r][t + 1]) {

                potentialLines.push_back({ A[r][t], {r, t} });
            }
        }
    }

    // Sort the top lines once after all peaks have been found
    sort(potentialLines.rbegin(), potentialLines.rend());
    for (int i = 0; i < min(3, static_cast<int>(potentialLines.size())); ++i) {
        topLines[i] = potentialLines[i];
    }

    for (int i = 0; i < 3; ++i) {
        cout << "Line " << i + 1 << ": ";
        cout << "Length = " << topLines[i].first << ", ";
        cout << "Rho = " << topLines[i].second.first << ", ";
        cout << "Theta = " << topLines[i].second.second << endl;
    }
    

    // Draw the top three lines
    for (int i = 0; i < 3; ++i) {
        // Get the top line from the top lines
        pair<unsigned int, pair<int, int>> top = topLines[i];

           // Draw the line
        int rho = top.second.first;
        int theta = top.second.second;
        double angle = theta ;
         double cosTheta = cos(angle);
         double sinTheta = sin(angle);
         for (int x = 0; x < COLS; ++x) {
             int y = static_cast<int>((rho - maxRho / 2) - x * (cosTheta / sinTheta));
             if (y >= 0 && y < ROWS && x >= 0 && x < COLS) {
                 // Ensure that the coordinates are within the image bounds
                 result[y][x] = 255; // Set pixel to white (edge)
             }
         }
     }


    
}



