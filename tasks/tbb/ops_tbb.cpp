#include "tbb/bychikhina_k_ring_tbb/include/ops_tbb.hpp"

#include <mpi.h>
#include <tbb/tbb.h>
#include <iostream>

#include "ring_topology.hpp"
#include <iostream>
#include <cstring>
#include <tbb/task_scheduler_init.h>

// Вычисляем соседей в кольце
RingTopology::RingTopology(int ws, int wr)
    : world_size(ws), world_rank(wr) 
{
    left  = (world_rank - 1 + world_size) % world_size;
    right = (world_rank + 1) % world_size;
}

// Метод передачи сообщения от source -> dest
void RingTopology::sendMessage(int source, int dest, const std::string& msg) {
    char buf[256];
    if (world_rank == source) {
        // Копируем сообщение в буфер
        std::strncpy(buf, msg.c_str(), sizeof(buf));
        // Отправляем правому соседу
        MPI_Send(buf, sizeof(buf), MPI_CHAR, right, 0, MPI_COMM_WORLD);
    } else {
        // Ждём сообщение от левого соседа
        MPI_Recv(buf, sizeof(buf), MPI_CHAR, left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (world_rank != dest) {
            // Если не адресат — пересылаем правому соседу
            MPI_Send(buf, sizeof(buf), MPI_CHAR, right, 0, MPI_COMM_WORLD);
        } else {
            // Конечный получатель выводит сообщение
            std::cout << "Процесс " << world_rank << " получил сообщение: " << buf << std::endl;
        }
    }
}

// -------------------------------------------
// Main
// -------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int threads_count = 1;
    int source = 0, dest = world_size - 1;

    if (world_rank == 0) {
        std::cout << "Введите число потоков TBB: ";
        std::cin >> threads_count;
        std::cout << "Введите ID источника: ";
        std::cin >> source;
        std::cout << "Введите ID получателя: ";
        std::cin >> dest;
    }

    // Рассылаем настройки всем процессам
    MPI_Bcast(&threads_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&source, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dest, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Инициализация TBB с указанным количеством потоков
    tbb::task_scheduler_init init(threads_count);

    // Создаём кольцевую топологию
    RingTopology ring(world_size, world_rank);

    // Отправляем сообщение
    ring.sendMessage(source, dest, "Hello from process " + std::to_string(source));

    MPI_Finalize();
    return 0;
}