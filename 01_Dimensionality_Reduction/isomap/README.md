# Steps of Isomap

1. Set a graph, $G$ with neighbors using $\text{k}$-graph

2. Find the shortest distance using Dijkstra algorithm

    Initialize:

    $$d_\mathbf{G}(i,j)=\begin{cases}
    d_\mathbf{x}(i,j)& \text{ if i,j are linked by an edge}\\
    \infty & \text{ otherwise }
    \end{cases}$$

    Update using Dijkstra algorithm

    $$\begin{aligned}
    d_\mathbf{G}(i,j) &= \text{min} 
    \begin{Bmatrix}
    d_\mathbf{G}(i,j), \,\, d_\mathbf{G}(i,k)+d_\mathbf{G}(k,i)
    \end{Bmatrix}\\
    k&=1,2,...N
    \end{aligned}$$

3. Find the vectors with a length of $d$ by MDS(In conclusion, it is same as using KernelPCA)

    * Goal: Find a new vector $y_i$ in $\mathbf{Y}$

    $$\begin{aligned}
    \text{Min }E &=||\tau (D_\mathbf{G})-\tau (D_\mathbf{Y})||_{L^2} \\
    \text{where} \,\,\, D_\mathbf{Y}&={ 
    ||\mathbf{y}_i-\mathbf{y}_j||
    } \,\,\,\, / \,\,\,\,
    ||*||_{L^2}=L^2 \, \text{matrix norm}=\sqrt{\sum_{i,j}{A^2_{ij}}} \\
    \tau (D) &=-HSH/2 \,\,\, \\
    where \,\,\, S&=\text{matrix of squared distances} = \begin{Bmatrix} S_{ij}=D^2_{ij} \end{Bmatrix} \\
    H&= \text{centering matrix}= \begin{Bmatrix}  \delta_{ij}-1/N \end{Bmatrix}
    \end{aligned}$$

    * $\lambda_{p}$: p-th eigenvalue of $\tau(D_G)$, $v_p^k$: k-th element of p-th eigenvector

    $$\therefore \text{p-th element of } y_i \in R^d = \sqrt{\lambda_p}v^i_p$$
