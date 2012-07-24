#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <map>

#define INTEGRAL_ITER	10
#define INPUT_BUFFER_SIZE       256

typedef double (* get_integral_value) (double, int);
typedef std::map<int, MPI_Request*> request_map_t;
typedef std::pair<int, MPI_Request *> request_entry_t;
typedef std::map<int, double *> recvbuf_map_t;
typedef std::pair<int, double *> recvbuf_entry_t;
double get_integr_value(double x, int func_number)
{
	double result = 0;
	//No need to omp parallel for, because this code is called from "omp parallel for" in integrateFunc
	for(int i = 0; i < func_number; ++i)
	{
		result += sin(x); 
	}

	return result; 
}

double integrate(double left_limit, double right_limit, int func_number, get_integral_value func)
{
	double result = 0;
	double step = (right_limit - left_limit) / INTEGRAL_ITER;

#pragma omp parallel for reduction(+:result) 
	for (int i = 0; i <= INTEGRAL_ITER; i++) 
	{
		result += func(left_limit + step * i, func_number) * step ;
	}

	return result;
}

int main(int argc, char* argv[])
{
	int rank, size;
	double count;
	double task[4]; 
	int task_id = 0;
	char input[INPUT_BUFFER_SIZE];
	double * recvbuf;
	MPI_Request	* recv_requests;

	recvbuf_map_t recvbuf_map;
	request_map_t recv_requests_map;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);

	if(0 == rank)
	{
		std::cout << "Hello! Lets start." << std::endl << 
			"Input limits and the number of integrate functions: Input data format: left_limit right_limit func_num." << std::endl <<
			"Example: 4 1.5 2" << std::endl <<
			"To quit type \"quit\"" <<std::endl;

		while(true)
		{
			std::cin.getline(input, sizeof(input));
			
			if(!strcmp(input, "quit"))
			{
				task_id = -1;
			}
			else
			{
				sscanf_s(input, "%lf %lf %lf", &task[0], &task[1], &task[2]);
				std::cout << "Data for task " << task_id << ": " << task[0] << " " << task[1] << " " << task[2] << std::endl; 
			}

			task[3] = task_id;

			MPI_Bcast(task, 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			
			if( -1 == task_id)//quit
			{
				//wait for all results
				for(request_map_t::iterator it = recv_requests_map.begin(); it != recv_requests_map.end(); ++it)
				{
					MPI_Waitall(size - 1, it->second, MPI_STATUS_IGNORE);
					
					double result = 0;
					for(int i = 1; i < size; ++i)
						result += recvbuf_map[it->first][i];
					std::cout << "Result for task " << it->first << ": " << result << std::endl; 
				
					delete[] it->second;
				}

				//cleanup
				for(recvbuf_map_t::iterator iter = recvbuf_map.begin(); iter != recvbuf_map.end(); ++iter)
					delete[] iter->second;	
					
				MPI_Finalize();
				return 0;
			}
			
			//prepair buffers for requests and results
			recvbuf = new double[size];
			recv_requests = new MPI_Request[size - 1];
			
			recvbuf_map.insert(recvbuf_entry_t(task_id, recvbuf));
			recv_requests_map.insert(request_entry_t(task_id, recv_requests));
			
			//send all tasks
			for(int i = 1; i < size; ++i)
				MPI_Irecv(&recvbuf[i], 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &recv_requests[i-1]);
			
			//check if any task is ready 
			request_map_t::iterator it = recv_requests_map.begin();
			while(it != recv_requests_map.end())
			{
				int flag = 0;

				MPI_Testall(size - 1, it->second, &flag, MPI_STATUS_IGNORE);
				
				if(flag)
				{
					int task = it->first;
					
					//print result
					double result = 0;
					for(int i = 1; i < size; ++i)
						result += recvbuf_map[task][i];
					std::cout << "Result for task " << task << ": " << result << std::endl; 
					
					//remove completed tasks
					delete[] it->second;
					recv_requests_map.erase(it++);

					recvbuf_map_t::iterator iter = recvbuf_map.find(task);
					delete[] iter->second;
					recvbuf_map.erase(iter);
				}
				else
				{
					++it;
				}
			}

			task_id++;
		}
	}
	else
	{
		while(true)
		{
			MPI_Bcast(task, 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);

			if(-1 == task[3])
			{
				std::cout << "Process " << rank << " quit." << std::endl; 
				MPI_Finalize();
				return 0;
			}

			//how many functions to calculate per node(based on user's input and available nodes)
			int amount = ((int)task[2] % (size - 1) ) ? ((int)task[2] / (size - 1)) : ((int)task[2] / (size - 1) + 1);
			
			if(0 == amount)
				amount++;
			
			count = 0;
			
			//each node calculate "amount" number functions. Start value is based on node's rank
			for(int i = 0; i < amount; ++i)
				if((rank - 1)*amount + i < task[2])
					count += integrate(task[0], task[1], (rank - 1)*amount + i + 1, get_integr_value);
			
			MPI_Send(&count, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
			std::cout << "Process " << rank << ": send result for task_id: " << task[3] <<  ". Result: " << count << std::endl;
		}
	}

	return 0;
}