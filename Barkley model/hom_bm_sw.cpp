#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

// Physical parameters
const double a = 0.3;
const double b = 0.01;
const double eps = 1.0 / 200.0;
const double gamma = 0.01;

// Numerical parameters
const int   L = 40;
const int   N = 81;
const double delta = 1e-4;
const double dt = (std::pow(L, 2)) / (5 * std::pow(N - 1, 2));
const double h = static_cast<double>(L) / (static_cast<double>(N) - 1);
const double one_m_dt = 1 - dt;
const double one_o_a = 1.0 / a;
const double b_o_a = b / a;
const double dt_o_eps = dt / eps;
const double D = dt / std::pow(h, 2);

// Indices for the laplacian fields used in alternation
int k = 0, kprm = 1;

// Data structures
std::vector<std::vector<double>> u(N + 1, std::vector<double>(N + 1, 0.0));
std::vector<std::vector<double>> v(N + 1, std::vector<double>(N + 1, 0.0));
std::vector<std::vector<double>> s(N + 1, std::vector<double>(N + 1, 0.0));
std::vector<std::vector<std::vector<double>>> lap(2, std::vector<std::vector<double>>(N + 2, std::vector<double>(N + 2, 0.0)));

// Function to initialize the spiral wave initial condition
void initialize_spiral_wave() {
	for (int i = 1; i <= N; ++i) {
		for (int j = 1; j <= N; ++j) {
			if (i < N / 2) {
				v[i][j] = a / 2.0;
			}
			if (j > N / 2) {
				u[i][j] = 1.0;
			}
		}
	}
}

// Function that initializes the heterogeneity
void initialize_heterogeneity() {

}

// Complete subroutine for taking one time step of the model
void step() {
	double u_th;

	// Interchange k and kprm
	int ktmp = kprm;
	kprm = k;
	k = ktmp;

	// Main loop
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			if (u[i][j] < delta) {
				u[i][j] = D * lap[k][i][j];
				v[i][j] = one_m_dt * v[i][j];
			}
			else {
				u_th = one_o_a * v[i][j] + b_o_a;
				v[i][j] = v[i][j] + dt * (u[i][j] - v[i][j]);
				// Implicit form for F
				if (u[i][j] < u_th)
					u[i][j] = u[i][j] / (1.0 - dt_o_eps * (1.0 - u[i][j]) * (u[i][j] - u_th)) + D * lap[k][i][j];
				else {
					double temp = dt_o_eps * u[i][j] * (u[i][j] - u_th);
					u[i][j] = (u[i][j] + temp) / (1.0 + temp) + D * lap[k][i][j];
				}
				lap[kprm][i][j] = lap[kprm][i][j] - 4 * u[i][j];
				lap[kprm][i + 1][j] = lap[kprm][i + 1][j] + u[i][j];
				lap[kprm][i - 1][j] = lap[kprm][i - 1][j] + u[i][j];
				lap[kprm][i][j + 1] = lap[kprm][i][j + 1] + u[i][j];
				lap[kprm][i][j - 1] = lap[kprm][i][j - 1] + u[i][j];
			}
			lap[k][i][j] = 0.0;
		}
	}
	// Impose no-flux boundary conditions
	for (int i = 1; i <= N; i++) {
		lap[kprm][i][1] = lap[kprm][i][1] + u[i][2];
		lap[kprm][1][i] = lap[kprm][1][i] + u[2][i];
		lap[kprm][i][N] = lap[kprm][i][N] + u[i][N - 1];
		lap[kprm][N][i] = lap[kprm][N][i] + u[N - 1][i];
	}
}

// Function to save the state of u and v to a single file
void save_state_to_file(std::ofstream& file) {
	if (!file) {
		std::cerr << "Error writing to file" << std::endl;
		return;
	}

	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			file << u[i][j] << " ";
		}
		file << "\n";
	}
	file << "#\n";
}

int main() {
	// Initialize the spiral wave initial condition
	initialize_spiral_wave();

	// Open file to save all steps
	std::ofstream file("simulation_data_barkley_model.txt");

	// Run the simulation for a certain number of steps and save the state
	int num_steps = 180;
	for (int step_num = 0; step_num < num_steps; ++step_num) {
		step();

		// Save the state
		if (step_num % 20 == 0) {
			save_state_to_file(file);
		}
	}

	// Close the file
	file.close();

	return 0;
}
