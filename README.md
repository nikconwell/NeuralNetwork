# NeuralNetwork
Experiment with some neural network stuff

Implements simple 2 layer network matrix for learning.  Works by adjusting weights of each node so that when given a specific input, it returns the same output.  Does not take advantage of multiple threads or a GPU so depending on your real world input this quickly gets to the point where things are taking a long time to learn.

Adjust the size and iteration in words3.py:
```Python
# Network of weights.  Sized to match input, in bits.  Middle layer 64 bits deep.
np.random.seed(1)
middle_size=64
w1 = 2*np.random.random((nn.__MAXSTRING__*8,middle_size))-1
w2 = 2*np.random.random((middle_size,nn.__MAXSTRING__*8))-1

# The bigger that middle_size gets, you will need to increase the iterations so that the
# network can stabilize.  There's probably some math with this, but I've just been experimenting randomly.
# I'm guessing these will have a nice chart if you can figure out optimal values.
for iter in range(100000):
```
You will find that as you increase the middle_size that you will need to greatly increase the range() so that the network will stabilize.

Feed the input to words3.py.  It saves the network in network.pickle.  Then use answer.py to get answers to questions.


Example usage in powershell

Our input data:
```powershell
gc -head 20 states.txt
```
```text
Alabama
Montgomery

Alaska
Juneau

Arizona
Phoenix

Arkansas
Little Rock

California
Sacramento

Colorado
Denver

Connecticut
Hartford
```

Do the learning:
```powershell
gc -head 20 states.txt | python words3.py
```
```text
0 errors (differences between input and output)
```

Here is the network:
```powershell
dir
```
```text
    Directory: C:\Users\Nik\Documents\GitHub\NeuralNetwork

Mode                LastWriteTime         Length Name
----                -------------         ------ ----
-a----       10/13/2019   5:35 AM         327888 network.pickle
```

Do some queries on the network:
```powershell
python answer.py
```
```text
Expecting lines of input (query) words on stdin.  ^D at end.
Arizona
output:  Phoenix
California
output:  Sacramento
Connecticut
output:  Hartford
^Z
```

If the network gets saturated, say if you run in a few more states on the basic network size:

```powershell
gc -head 26 states.txt | python words3.py
```
```text
2 errors (differences between input and output)
```
You will see slight errors on the output:
```powershell
python answer.py
```
```text
Connecticut
output:  Hartfopd
```
Notice the r was mis-learned or mis-remembered as a p.  Off by a bit.

