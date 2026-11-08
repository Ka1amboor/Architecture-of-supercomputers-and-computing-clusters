#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <mpi.h>

using namespace std;

struct Derivative {
    double dy_dx;
    double dz_dx;
};

Derivative f(double x, double y, double z) {
    Derivative d;
    d.dy_dx = z;
    d.dz_dx = -2.0 * z * (1.0 / tan(x)) - 3.0 * y;
    return d;
}

double exact_solution(double x) {
    return (-0.9783 * cos(2.0 * x) + 0.4776 * sin(2.0 * x)) / sin(x);
}

void runge_kutta_step(double x, double y, double z, double h, double& y_next, double& z_next) {
    Derivative k1 = f(x, y, z);
    Derivative k2 = f(x + h / 2.0, y + h * k1.dy_dx / 2.0, z + h * k1.dz_dx / 2.0);
    Derivative k3 = f(x + h / 2.0, y + h * k2.dy_dx / 2.0, z + h * k2.dz_dx / 2.0);
    Derivative k4 = f(x + h, y + h * k3.dy_dx, z + h * k3.dz_dx);

    y_next = y + h * (k1.dy_dx + 2.0 * k2.dy_dx + 2.0 * k3.dy_dx + k4.dy_dx) / 6.0;
    z_next = z + h * (k1.dz_dx + 2.0 * k2.dz_dx + 2.0 * k3.dz_dx + k4.dz_dx) / 6.0;
}

void adams_method(const vector<double>& x_values, const vector<double>& y_rk, const vector<double>& z_rk, double h) {
    int n = x_values.size();
    vector<double> y_values = y_rk; 
    vector<double> z_values = z_rk;

    cout << "\nМетод Адамса\n";
    cout << left << setw(6) << "Шаг" << setw(10) << "x" << setw(15) << "y_h" 
         << setw(20) << "y_h (испр)" << setw(15) << "Ошибка" << setw(15) << "z" << setw(15) << "Результат\n";
    cout << string(96, '-') << "\n";

    cout << left << setw(6) << 0 << setw(10) << fixed << setprecision(2) << x_values[0] 
         << setw(15) << setprecision(6) << y_values[0] << setw(20) << "-" 
         << setw(15) << "-" << setw(15) << z_values[0] << setw(15) << exact_solution(x_values[0]) << "\n";

    for (int i = 1; i < 4; i++) {
        runge_kutta_step(x_values[i - 1], y_values[i - 1], z_values[i - 1], h, y_values[i], z_values[i]);
        
        cout << left << setw(6) << i << setw(10) << fixed << setprecision(2) << x_values[i]
             << setw(15) << setprecision(6) << y_values[i] << setw(20) << "-" 
             << setw(15) << "-" << setw(15) << z_values[i] << setw(15) << "-" << "\n";
    }

    for (int i = 4; i < n; i++) {
        double y_h = y_values[i - 1] + h * (55.0 * z_values[i - 1] - 59.0 * z_values[i - 2] + 37.0 * z_values[i - 3] - 9.0 * z_values[i - 4]) / 24.0;
        double z_h = z_values[i - 1] + h * (
            55.0 * f(x_values[i - 1], y_values[i - 1], z_values[i - 1]).dz_dx
            - 59.0 * f(x_values[i - 2], y_values[i - 2], z_values[i - 2]).dz_dx
            + 37.0 * f(x_values[i - 3], y_values[i - 3], z_values[i - 3]).dz_dx
            - 9.0 * f(x_values[i - 4], y_values[i - 4], z_values[i - 4]).dz_dx
        ) / 24.0;
        double y_h_corrected = y_values[i - 1] + h * (9.0 * z_h + 19.0 * z_values[i - 1] - 5.0 * z_values[i - 2] + z_values[i - 3]) / 24.0;
        double z_h_corrected = z_values[i - 1] + h * (
            9.0 * f(x_values[i - 1] + h, y_h, z_h).dz_dx
            + 19.0 * f(x_values[i - 1], y_values[i - 1], z_values[i - 1]).dz_dx
            - 5.0 * f(x_values[i - 2], y_values[i - 2], z_values[i - 2]).dz_dx
            + f(x_values[i - 3], y_values[i - 3], z_values[i - 3]).dz_dx
        ) / 24.0;
        double error = abs(y_h_corrected - y_h) / (pow(2, 4) - 1);
        y_values[i] = y_h_corrected;
        z_values[i] = z_h_corrected;

        cout << left << setw(6) << i << setw(10) << fixed << setprecision(2) << x_values[i]
             << setw(15) << setprecision(6) << y_h << setw(20) << y_h_corrected 
             << setw(15) << setprecision(12) << error << setw(15) << setprecision(6) << z_values[i] 
             << setw(15) << exact_solution(x_values[i]) << "\n";
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double x0 = 1.0, y0 = 1.0, z0 = 1.0;
    double h = 0.1;
    double x_end = 2.0;
    int n = static_cast<int>((x_end - x0) / h) + 1;

    vector<double> x_values(n);
    vector<double> y_euler(n, 0.0), z_euler(n, 0.0);
    vector<double> y_rk(n, 0.0), z_rk(n, 0.0);
    vector<double> y_exact(n, 0.0);

    if (rank == 0) {
        for (int i = 0; i < n; i++) {
            x_values[i] = x0 + i * h;
            y_exact[i] = exact_solution(x_values[i]);
        }
    }

    MPI_Bcast(x_values.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(y_exact.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    y_euler[0] = y0; z_euler[0] = z0;
    y_rk[0] = y0; z_rk[0] = z0;

    // Считаем первые 3 точки синхронно
    for (int i = 1; i <= 3; i++) {
        Derivative d = f(x_values[i - 1], y_euler[i - 1], z_euler[i - 1]);
        y_euler[i] = y_euler[i - 1] + h * d.dy_dx;
        z_euler[i] = z_euler[i - 1] + h * d.dz_dx;
        runge_kutta_step(x_values[i - 1], y_rk[i - 1], z_rk[i - 1], h, y_rk[i], z_rk[i]);
    }

    int points_to_distribute = n - 4;
    int chunk = points_to_distribute / size;
    int remainder = points_to_distribute % size;
    
    int start_idx = 4 + rank * chunk + min(rank, remainder);
    int end_idx = start_idx + chunk + (rank < remainder ? 1 : 0);

    // --- КОНВЕЙЕР: Получение данных от предыдущего процесса ---
    if (rank > 0) {
        // Ждем данные от процесса rank - 1
        MPI_Recv(&y_euler[start_idx - 1], 2, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&z_euler[start_idx - 1], 2, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&y_rk[start_idx - 1], 2, MPI_DOUBLE, rank - 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&z_rk[start_idx - 1], 2, MPI_DOUBLE, rank - 1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Каждый процесс считает свой кусок
    for (int i = start_idx; i < end_idx; i++) {
        Derivative d = f(x_values[i - 1], y_euler[i - 1], z_euler[i - 1]);
        y_euler[i] = y_euler[i - 1] + h * d.dy_dx;
        z_euler[i] = z_euler[i - 1] + h * d.dz_dx;
        runge_kutta_step(x_values[i - 1], y_rk[i - 1], z_rk[i - 1], h, y_rk[i], z_rk[i]);
    }

    // --- КОНВЕЙЕР: Отправка данных следующему процессу ---
    if (rank < size - 1) {
        // Отправляем наш последний посчитанный узел следующему процессу
        // end_idx - 1 это индекс последней точки, которую мы посчитали.
        // Следующему процессу нужны значения ИЗ этой точки, чтобы начать с end_idx.
        MPI_Send(&y_euler[end_idx - 1], 2, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
        MPI_Send(&z_euler[end_idx - 1], 2, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD);
        MPI_Send(&y_rk[end_idx - 1], 2, MPI_DOUBLE, rank + 1, 2, MPI_COMM_WORLD);
        MPI_Send(&z_rk[end_idx - 1], 2, MPI_DOUBLE, rank + 1, 3, MPI_COMM_WORLD);
    }

    // --- СБОР РЕЗУЛЬТАТОВ ---
    // (Точно такой же, как в прошлый раз, но теперь у нас правильные данные)
    vector<int> recvcounts(size), displs(size);
    for (int i = 0; i < size; i++) {
        recvcounts[i] = points_to_distribute / size + (i < remainder ? 1 : 0);
        displs[i] = 4 + i * chunk + min(i, remainder); 
    }

    vector<double> send_y_euler(y_euler.begin() + start_idx, y_euler.begin() + end_idx);
    vector<double> send_z_euler(z_euler.begin() + start_idx, z_euler.begin() + end_idx);
    vector<double> send_y_rk(y_rk.begin() + start_idx, y_rk.begin() + end_idx);
    vector<double> send_z_rk(z_rk.begin() + start_idx, z_rk.begin() + end_idx);

    MPI_Allgatherv(send_y_euler.data(), end_idx - start_idx, MPI_DOUBLE,
                   y_euler.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);
    MPI_Allgatherv(send_z_euler.data(), end_idx - start_idx, MPI_DOUBLE,
                   z_euler.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);
    MPI_Allgatherv(send_y_rk.data(), end_idx - start_idx, MPI_DOUBLE,
                   y_rk.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);
    MPI_Allgatherv(send_z_rk.data(), end_idx - start_idx, MPI_DOUBLE,
                   z_rk.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    // --- ВЫВОД РЕЗУЛЬТАТОВ ---
    if (rank == 0) {
        cout << "\nEuler's method\n";
        cout << "x          y_euler       y'_euler      y_exact      ε_euler\n";
        cout << "-----------------------------------------------------------\n";
        for (int i = 0; i < n; i++) {
            double err_euler = abs(y_exact[i] - y_euler[i]) / (pow(2, 2) - 1);
            cout << fixed << setprecision(8) << x_values[i] << "  " 
                 << y_euler[i] << "  " << z_euler[i] << "  " 
                 << y_exact[i] << "  " << err_euler << "\n";
        }

        cout << "\nRunge-Kutta method\n";
        cout << "x          y_runge_kutta  y'_runge     y_exact      ε_runge_kutta\n";
        cout << "-------------------------------------------------------------------\n";
        for (int i = 0; i < n; i++) {
            double err_rk = abs(y_exact[i] - y_rk[i]) / (pow(2, 4) - 1);
            cout << fixed << setprecision(8) << x_values[i] << "  " 
                 << y_rk[i] << "  " << z_rk[i] << "  " 
                 << y_exact[i] << "  " << err_rk << "\n";
        }

        adams_method(x_values, y_rk, z_rk, h);

        cout << "\nSolution by Euler's method\n";
        cout << "---------------------------\n";
        cout << "x_values: [";
        for (int i = 0; i < n; i++) cout << x_values[i] << (i < n - 1 ? ", " : "");
        cout << "]\n";
        cout << "y_euler: [";
        for (int i = 0; i < n; i++) cout << y_euler[i] << (i < n - 1 ? ", " : "");
        cout << "]\n";

        cout << "\nSolution by Runge-Kutta method\n";
        cout << "-------------------------------\n";
        cout << "x_values: [";
        for (int i = 0; i < n; i++) cout << x_values[i] << (i < n - 1 ? ", " : "");
        cout << "]\n";
        cout << "y_runge_kutta: [";
        for (int i = 0; i < n; i++) cout << y_rk[i] << (i < n - 1 ? ", " : "");
        cout << "]\n";
    }

    MPI_Finalize();
    return 0;
}