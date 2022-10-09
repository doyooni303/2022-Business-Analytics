# Steps of locally linear embedding

1. Assign k, which is a parameter, negihbors to each data point $\overrightarrow{x}_i$

2. Reconstruct with linear weights( $w_{ij}$ )

    $$\begin{aligned}
    \text{Min} \quad &E(W_i)=|\overrightarrow{x}_i-\sum_{j}{W_{ij}\overrightarrow{x}_j}|^2\\
    s.t \quad &W_{ij}=0\quad \text{if }\quad \overrightarrow{x}_j \quad \text{does not belong to the neighbors of} \quad \overrightarrow{x}_i \\
    &\sum_{j}{W_{ij}=1 ,\forall i} 
    \end{aligned}$$

    [Solution]

    $$\begin{aligned}
    E_i(W)&=|\overrightarrow{x}_i-\overrightarrow{x}_i-\sum_{j}{W_{ij}\overrightarrow{x}_j}+\overrightarrow{x}_i|^2\\
    &=|-\sum_{j}{W_{ij}\overrightarrow{x}_j}+\sum_{j}{W_{ij}}\overrightarrow{x}_i|^2\\
    &=|\sum_{j}{W_{ij}}(\overrightarrow{x}_i-\overrightarrow{x}_j)|^2 \\
    &=|\sum_{j}{W_{ij}}\overrightarrow{z}_j|^2 \quad (\overrightarrow{z}_j=\overrightarrow{x}_j-\overrightarrow{x}_i) \\
    &=(Z^TW_i)^T(Z^TW_i)\\
    &=W_i^TZZ^TW_i \\
    &=W_i^TG_iW_i \quad (G_i=ZZ^T,\text{Gram Matrix}) \\
    \end{aligned}$$

    The optimization problem can be rewritten as follows.

    $$\begin{aligned}
    \text{Min} \quad &E_i(W)=W_i^TG_iW_i \\
    \text{s.t.} \quad &\sum_{j}{w_{ij}=1}=\mathbf{1}^TW_i \\
    \end{aligned}$$

    By Lagrangian multiplier and its KKT condition,
    
    $$\begin{aligned}
    L(W_i,\lambda)&=W^T_iG_iW_i-\lambda(\mathbf{1}^TW_i-1) \\
    \frac{\delta L}{\delta W_i}&=(G_i+G^T_i)W_i-\lambda \mathbf{1} \quad(\because \text{G is symmetric})\\
    &=2G_iW_i-\lambda \mathbf{1}=0 \\
    \therefore W_i&=\frac{\lambda}{2}G_i^{-1}\mathbf{1}\\
    \end{aligned}$$

3. Map to embedded coordinates

    $$\begin{aligned}
    \text{Min} \quad \Phi(Y)&=\sum_{i}{|\overrightarrow{y}_i-\sum_{j}{w_{ij}\overrightarrow{y}_j}|^2}\\
    \text{s.t}&\sum_{i}{\overrightarrow{y}_i}=0 \\
    &\frac{1}{n}Y^TY=I \quad(n\text{ is the number of samples})
    \end{aligned}$$
    
    By using the conditions, 

    $$\Phi(Y)=Y^TMY \quad \text{where} \quad M=(I-W)^T(I-W)$$

    Therefore, the problem can be rewritten as follows.

    $$\begin{aligned}
    \text{Min} \quad \Phi(Y)&=Y^TMY\\
    \text{s.t}&\sum_{i}{\overrightarrow{y}_i}=0 \\
    &\frac{1}{n}Y^TY=I \quad(n\text{ is the number of samples})
    \end{aligned}$$

    By Lagrangian multiplier and its KKT condition,

    $$\begin{aligned}
    L(Y,\alpha)&=Y^MY-\alpha(\frac{1}{n}Y^TY-I) \\
    \frac{\delta L}{\delta Y}&=(Y+Y^T)M-\frac{2}{m}\alpha Y \\
    &=2MY-\frac{2}{m}\alpha Y=0\\
    \therefore MY=\frac{\alpha}{n}Y
    \end{aligned}$$