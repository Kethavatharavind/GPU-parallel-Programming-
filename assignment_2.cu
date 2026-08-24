#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <cuda_runtime.h>
using namespace std;

vector<int> sssp(vector<vector<pair<int,int>>> &graph,int src)
{
    int n=graph.size();
    vector<int> dist(n,999999);
    dist[src]=0;

    priority_queue<pair<int,int>,vector<pair<int,int>>,greater<pair<int,int>>> pq;
    pq.push({0,src});

    while(!pq.empty())
    {
        int d=pq.top().first;
        int u=pq.top().second;
        pq.pop();

        for(auto edge:graph[u])
        {
            int v=edge.first;
            int w=edge.second;

            if(d+w<dist[v])
            {
                dist[v]=d+w;
                pq.push({dist[v],v});
            }
        }
    }
    return dist;
}

__global__ void ssspkernel(int n,int *row,int *col,int *wt,int *dist,int *changed)
{
    int u=blockIdx.x*blockDim.x+threadIdx.x;

    if(u>=n)
        return;

    if(dist[u]==999999)
        return;

    for(int i=row[u];i<row[u+1];i++)
    {
        int v=col[i];
        int w=wt[i];

        if(dist[u]+w<dist[v])
        {
            atomicMin(&dist[v],dist[u]+w);
            atomicExch(changed,1);
        }
    }
}

vector<int> gpu_sssp(int n,vector<int> &row,vector<int> &col,vector<int> &wt,int src)
{
    int m=col.size();

    vector<int> dist(n,999999);
    dist[src]=0;

    int *drow,*dcol,*dwt,*ddist,*dchanged;

    cudaMalloc(&drow,(n+1)*sizeof(int));
    cudaMalloc(&dcol,m*sizeof(int));
    cudaMalloc(&dwt,m*sizeof(int));
    cudaMalloc(&ddist,n*sizeof(int));
    cudaMalloc(&dchanged,sizeof(int));

    cudaMemcpy(drow,row.data(),(n+1)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(dcol,col.data(),m*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(dwt,wt.data(),m*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(ddist,dist.data(),n*sizeof(int),cudaMemcpyHostToDevice);

    int threads=256;
    int blocks=(n+threads-1)/threads;

    int changed=1;

    while(changed)
    {
        changed=0;

        cudaMemcpy(dchanged,&changed,sizeof(int),cudaMemcpyHostToDevice);

        ssspkernel<<<blocks,threads>>>(n,drow,dcol,dwt,ddist,dchanged);
        cudaDeviceSynchronize();

        cudaMemcpy(&changed,dchanged,sizeof(int),cudaMemcpyDeviceToHost);
    }

    cudaMemcpy(dist.data(),ddist,n*sizeof(int),cudaMemcpyDeviceToHost);

    cudaFree(drow);
    cudaFree(dcol);
    cudaFree(dwt);
    cudaFree(ddist);
    cudaFree(dchanged);

    return dist;
}

int main()
{
    ifstream file("graph.csr");

    int n,m;
    file>>n>>m;

    vector<int> row(n+1);
    vector<int> col(m);
    vector<int> wt(m);

    for(int i=0;i<=n;i++)
        file>>row[i];

    for(int i=0;i<m;i++)
        file>>col[i];

    for(int i=0;i<m;i++)
        file>>wt[i];

    file.close();

    vector<vector<pair<int,int>>> graph(n);

    for(int u=0;u<n;u++)
        for(int i=row[u];i<row[u+1];i++)
            graph[u].push_back({col[i],wt[i]});

    int src=0;

    vector<int> cpuresult=sssp(graph,src);
    vector<int> gpuresult=gpu_sssp(n,row,col,wt,src);

    cout<<"cpu result:"<<endl;
    for(int i=0;i<n;i++)
        cout<<"vertex "<<i<<" = "<<cpuresult[i]<<endl;

    cout<<endl;

    cout<<"gpu result:"<<endl;
    for(int i=0;i<n;i++)
        cout<<"vertex "<<i<<" = "<<gpuresult[i]<<endl;

    return 0;
}