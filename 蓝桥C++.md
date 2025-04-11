## 一.STL

## 1.栈 (先进先出)

```cpp
头文件：stack

定义：stack<T> s;

压入：s.push(x); //栈尾 O(1)

弹出：s.pop(); //栈尾O(1)

查看栈顶元素：s.top(); //栈尾 O(1)

查看栈里面的元素个数：s.size(); //O(1)
```



## 2.队列

```cpp
头文件：queue

定义：queue<T> q;

入队：q.push(x); //队尾 O(1) 

出队：q.pop(); //队首 O(1)

查看队首元素：q.front(); //O(1)

查看队列中的元素个数：q.size(); //O(1)
```



### 2.1 双端队列

```cpp
头文件：queue
定义：deque<T> q;
队尾push：q.push_back();
队首push：q.push_front();
队尾pop：q.pop_back();
队首pop：q.pop_front();
查看队尾：q.back();
查看队首：q.front();
复杂度均为O(1)
```

## 2.2优先队列（大根堆）

```cpp
头文件：queue
定义:priority_queue<T> q;
插入：q.push();
删除最大值：q.pop(); //O(logn)
查看最大值：q.top(); //O(1)
查看元素个数：q.size(); //O(1)
```

## 3.向量



```cpp
头文件：vector

定义：vector<T> a;

尾部加入一个元素：a.push_back(x); //O(1)

查看元素个数：a.size(); //O(1)

查看第i个元素：a[i];//O(1)

遍历：for(int i=0;i<a.size();i++) cout<<a[i]; 
```



## 4.排序

```cpp
从小到大
sort(a,a+n);

从大到小
bool cmp(int x,int y){
returnx>y;

}
sort(a,a+n,cmp);
```



## 5.字符串

```cpp
头文件：string
定义：string s;
长度：s.size();//O(1)
第i个字符（下标从0开始到size-1）:s[i];//O(1)
翻转：s.reverse();//O(n)
赋值：s=a;//O(n),a可以是一个string，一个char数组或char类型
在后面加一个字符或字符串：s+=a;//O(a的长度),a的类型同上
删除：s.erase(i,2);//从i到i+1都删除
插入、删除、替换、查找、取子串、比较：insert、erase、replace、find、substr、compare
//参考： https://legacy.cplusplus.com/reference/string/string/?kw=string
按字典序比较：使用比较符号即可
连接两个字符串：s1+s2
输入：cin>>s;
输出：cout<<s;
输入一行：getline(cin,s);
```



## 6.集合

```cpp
该集合与数学中的集合定义类似，不存在相等的元素。
即不存在元素x与y，使得x<y与y<x同时不成立。
set使用红黑树实现，是一个平衡树。有序
头文件：set
定义：set<T> s;
插入：s.insert(x);//O(logn)
删除：s.erase(x);//O(logn)，x可以为值或者迭代器
元素个数：s.size();//O(1)
清空集合：s.clear();//O(n)
前一个元素：prev(it);//O(1)
后一个元素：next(it);//O(1)
使迭代器向后移动一个元素：++it;//O(1)
使迭代器向前移动一个元素：--it;//O(1)
从小到大遍历：for(set<T>::iterator it=s.begin();it!=s.end();++it) cout<<*it<<endl;//O(n)
从大到小遍历：for(set<T>::reverse_iterator it=s.rbegin();it!=s.rend();++it)//O(n)
查找：s.find(x);//O(logn)，若找到了则会返回相应元素的迭代器，若没找到则会返回s.end()，下同
查找大于等于x的第一个元素：s.lower_bound(x);//O(logn)
查找大于x的第一个元素：s.upper_bound(x);//O(logn)
s.begin() 返回一个指向 std::set 容器中第一个元素的迭代器。
s.end() 返回一个指向 std::set 容器中最后一个元素之后位置的迭代器。
```



## 7.映射（字典）

```cpp
map是一个值到另一个值的映射，类似python里的字典。  
map 容器会根据键的字典序自动对元素进行排序。 比如我输入的键是4，3，2，7，输出的键则是2，3，4，7
每一对值分为键key和值value，以key当做查找的关键字。
map也是用红黑树实现的，各个操作复杂度同set
头文件：map
定义：map <T1,T2> a;
赋值：a[x]=y;;//O(logn)若已有键x，那么将其对应的值改成y
//否则插入一对新的键值x和y
取x对应的值：a[x];//O(logn)
查找是否存在键x：a.find(x);O(logn)//若找到则会返回对应迭代器，否则返回a.end()
//若想看键x是否存在，一定要用find，而不要用中括号
查看该迭代器的键：it ->first;
查看该迭代器的值：it->second;
删除元素：a.erase(it);//O(1)用法同set
```

## 8.pair 一对

```cpp
 pair是一个很实用的"小玩意",当想要将两个元素绑在一起作为一个合成元素,又不想因此定义结构体时,使用pair可以很方便地作为一个替代品。
 #define x first
 #define y second
 typedef pair<int,int> PII; 用PII代替pair<int,int>
 PII a;
 a.x=3;
 a.y=5;

```

```cpp
//如果想要在代码中临时构建一个pair,有如下两种方法:
   // (1)将类型定义写在前面，后面用小括号内两个元素的方式。
pair<string,int>("haha",5);
//    (2)使用自带的make_pair函数
	make_pair("haha",5);
```



# 二.零散知识

## 1.字符串大写改为小写tolower(s[i]);

```cpp
string s = "HELLO WORLD";
for (int i = 0; i < s.size(); i++) {
s[i] = tolower(s[i]);
}
```

## 2.getline()

```cpp
string m;
getline(cin,m,"#"); //第三个参数是结束符，默认回车为结束符
```

在使用getline之前，如果已经使用了cin来读取数据，需要注意cin可能会在输入缓冲区中留下一个换行符。如果不处理这个换行符，getline可能会立即返回一个空字符串，因为它认为已经读到了一行的结束。为了避免这个问题，可以在调用getline之前使用另一个getline来消耗掉这个换行符

```cpp
cin >> x; // 读取一个整数
cin.ignore(); // 忽略换行符 也可以多使用一次getline(cin,str)来代替
getline(cin, str); // 然后使用getline读取一行文本
```

## 3.计算n阶乘后的数字的末尾0的个数的正常方法

**（10的阶乘是3628800，末尾0的个数是2）**

```
ll jie(ll n){
int f1=1;
   for(int i=1;i<=n;i++){
   f1*=i;
   }
   //最终计算出来的f1就是阶乘
   int count=0;
   while(f1%10==0&&f1!=0){ //f1的末尾有0并且f1不为0
   count++;
   f1/=10; //取走末尾的一个0
   }
    return count;
}

```



## 4.计算n阶乘后的数字的末尾0的个数的巧妙法

```cpp
ll jie(ll n){
    ll count = 0;
    //n本身可以被5整除，加整除后n的值，阶乘后的末尾就有几个零
    while (n >= 5) {
        n /= 5;
        count += n;
    }
    return count;
}
```

改良版本：用map字典存计算过的n

```cpp
map<ll,ll>memo;
ll jie(ll n){
    //C++11:  auto it=memo.find(n);
    map<ll,ll>::interator it=memo.find(n);
    if(it!=memo.end()) return memo[n];
    
    ll count = 0;
    //n本身可以被5整除几次，阶乘后的末尾就有几个零
    while (n >= 5) {
        n /= 5;
        count += n;
    }
    memo[n] = count;
    return count;
}
```

## 5.ASICII

![ASICII](assets/ASICII.png)

## 6.bit Byte

![bit Byte](assets/bit Byte.png)

## 7.printf

![printf](assets/printf.png)

## 8.int_long

![int_long](assets/int_long.png)

## 9.倍数-约数

![倍数-约数](assets/倍数-约数.png)

## 10.string和char

### string

```cpp
string a;
cin>>a; //a=ffgwwgr23234;
string b[10];
for(int i=0;i<10;i++) cin>>b[i]; //可以cout<<b[2][3];
```

### char

```cpp
char a;
cin>>a; //只能是一个字符2 单引号''
char a[10];
cin>>a; //可以输入十个字符
char b[10][5];
for(int i=0;i<10;i++){
cin>>b[i]; //输入十次，每次可以输入5个字符
}
```

## 11.结构体

```
struct Log{
	int ts;
	int id;
}log[1000];
bool cmp(const Log& a,const Log& b){
return a.ts>b.ts;  //从大到小排序
}
sort(log,log+1000,cmp);
```



# 三. 前缀和

就是在cin>>a[i]的同时，计算每一次的和 即

```c++
for( int i=1;i<=n;i++){

cin>>a[i];

sum[i] = sum[i-1]+a[i];}
```

## 1.一维前缀和

![image-20250409112218187](assets/image-20250409112218187.png)

```c++
#include<bits/stdc++.h>
using namespace std;
const int N=100010;
int s[N],t[N];
int n,m;
int main(){
    scanf("%d %d",&n,&m);
    for(int i=1;i<=n;i++){
        scanf("%d",&t[i]);
        s[i]=s[i-1]+t[i];
    }
    int l,r;
    while(m--){
        scanf("%d %d",&l,&r);
        printf("%d\n",s[r]-s[l-1]);
    }
    
}
```



## 2.二维前缀和

![image-20250409112319727](assets/image-20250409112319727.png)

```C++
#include <iostream>
using namespace std;
const int N=1010;
int sum[N][N];
int main()
{
  int n,m,q,h;
  cin>>n>>m>>q;
  for(int i=1;i<=n;i++){
    for(int j=1;j<=m;j++){
      cin>>h;
      sum[i][j] = sum[i-1][j]+sum[i][j-1]+h-sum[i-1][j-1]; 从（1，1）到（i,j)的数的和
    }
  }
  int x1,x2,y1,y2;
  while(q--){
    cin>>x1>>y1>>x2>>y2;
    int ans = sum[x2][y2]-sum[x1-1][y2]-sum[x2][y1-1]+sum[x1-1][y1-1];
    cout<<ans<<endl;
  }
  return 0;
}
```

# 四.差分

## 1.一维差分

![image-20250410221914191](assets/image-20250410221914191.png)

![image-20250410221924587](assets/image-20250410221924587.png)

![e13c3da71d36f8f24f76c41ef4f4cb8](assets/e13c3da71d36f8f24f76c41ef4f4cb8.jpg)

![515f29f4a928a78602ddfe16ecb3e41](assets/515f29f4a928a78602ddfe16ecb3e41.jpg)

## 2.二维差分

```cpp
#include <iostream>
using namespace std;
const int N=1010;
int ans[N][N],dd[N][N];
int main()
{
  int n,m,q;
  cin>>n>>m>>q;
  for(int i=1;i<=n;i++){
    for(int j=1;j<=m;j++){
    cin>>ans[i][j];
    dd[i][j]=ans[i][j]-ans[i-1][j]-ans[i][j-1]+ans[i-1][j-1];
  }}
  int x1,x2,y1,y2,d;
  while(q--){
    cin>>x1>>y1>>x2>>y2>>d;
    dd[x1][y1]+=d;
    dd[x1][y2+1]-=d;
    dd[x2+1][y1]-=d;
    dd[x2+1][y2+1]+=d;
  }
   for(int i=1;i<=n;i++)
    {
      for(int j=1;j<=m;j++){
        ans[i][j]=dd[i][j]+ans[i-1][j]+ans[i][j-1]-ans[i-1][j-1];
        cout<<ans[i][j]<<" ";
      }
      cout<<endl;
    }
  return 0;
}
```



# 五.二分

![Screenshot_20250310_203332_com.huawei.browser](assets/Screenshot_20250310_203332_com.huawei.browser.jpg)

## 1.二分找最大值

```cpp
while(l<r){
int mid=(l+r+1)/2
if(mid<=ans) l=mid;
else r=mid-1;
}
```

## 2.二分找最小值

```cpp
while(l<r){
int mid(l+r)/2;
if(mid>=ans) r=mid;
else l=mmid+1;
}
```

## 3.注意边界影响，比如：

```cpp
int l=1,r=MAX_INT;
int mid=(l+r)/2; //l+r的结果超过了int类型的表示范围，溢出了，可以让l=1,r=MAX_INT-1
```



# 六.动态规划 DP

## 1.闫式DP分析法 

#### f[i] [j]表示的是集合的属性，是一个值，比如在01背包问题里表示集合中的最大值 i表示物品的数量，j表示体积

![Snipaste_2025-03-19_11-30-15](assets/Snipaste_2025-03-19_11-30-15.png)

## 2. 01背包问题

![image-20250411191419147](assets/image-20250411191419147.png)

```cpp
#include<bits/stdc++.h>
using namespace std;
int v[1010],w[1010];
int f[1010][1010];
int main()
{
    int n,m;
    cin>>n>>m;
    for(int i=1;i<=n;i++){
        cin>>v[i]>>w[i];
    }
    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++){
            f[i][j]=f[i-1][j];
            if(j>=v[i]) f[i][j]=max(f[i][j],f[i-1][j-v[i]]+w[i]);
        }
    }
    cout<<f[n][m]<<endl;
    return 0;
}
```

