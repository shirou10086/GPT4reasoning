'''
互通设备集

同一局域网内的设备可以相互发现，具备直连路由的两个设备可以互通。假定设备A和B互通，B和C互通，那么可以将B作为中心设备，通过多跳路由策略使设备A和C互通。这样，A、B、C三个设备就组成了一个互通设备集。其中，互通设备集包括以下几种情况：
1）直接互通的多个设备
2）通过多跳路由策略间接互通的多个设备
3）没有任何互通关系的单个设备
现给出某一局域网内的设备总数以及具备直接互通关系的设备，请计算该局域网内的互通设备集有多少个？

解答要求
时间限制: 1000ms, 内存限制: 256MB
输入
第一行：某一局域网内的设备总数M，32位有符号整数表示。1 <= M <= 200
第二行：具备直接互通关系的数量N，32位有符号整数表示。0 <= N < 200
第三行到第N+2行：每行两个有符号32位整数，分别表示具备直接互通关系的两个设备的编号，用空格隔开。每个设备具有唯一的编号，0 <= 设备编号 < M

输出
互通设备集的数量，32位有符号整数表示。

样例1
输入：
3总设备
2互通关系
0 1具体链接
0 2
输出：
1总互通数据集
解释：
编号0和1以及编号0和2的设备直接互通，编号1和2的设备可通过编号0的设备建立互通关系，互通设备集可合并为1个。

样例2
输入：
2
0
输出：
2
解释：
2个设备均不互通，返回2。

样例3
输入：
5
2
0 1
2 3
输出：
2
解释：
编号0和1的设备直接互通，编号2和3的设备直接互通，剩余编号4的设备与其他设备均无互通关系，返回3。'''
numdevice=5
connection=[(0,1),(2,3)]

def dfs(node,graph,visited):#深度查找连通数量
    if visited[node]:#如果是true,break
        return
    visited[node] =True
    for connection in graph[node]:
        dfs(connection,graph,visited)

def findcentral(numdevice,connection):
    map=[[] for i in range(numdevice)]#建立一个图[[],[],[]]
    for connectnode1,connectnode2 in connection:#[(0,1),(0,2)]
        map[connectnode1].append(connectnode2)
        map[connectnode2].append(connectnode1)#[[1],[0],[]]
    visitlist=[False]*numdevice#[[F],[F],[F]]
    ans=0
    for i in range(numdevice):
        if not visitlist[i]:
            dfs(i,map,visitlist)
            ans+=1
    return ans


print(findcentral(numdevice,connection))
'''假设输入：
numdevice=3
connection=[(0,1),(0,2)]

'''
