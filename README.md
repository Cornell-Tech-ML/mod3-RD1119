# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py


# Task 1 & 2
```bash
   python3 project/parallel_check.py
   ```

   ```bash
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/users/rundong/github/mod3-rd1119/minitorch/fast_ops.py (175)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /users/rundong/github/mod3-rd1119/minitorch/fast_ops.py (175)
-----------------------------------------------------------------------------|loop #ID
    def _map(                                                                |
        out: Storage,                                                        |
        out_shape: Shape,                                                    |
        out_strides: Strides,                                                |
        in_storage: Storage,                                                 |
        in_shape: Shape,                                                     |
        in_strides: Strides,                                                 |
    ) -> None:                                                               |
        # TODO: Implement for Task 3.1.                                      |
        out_size = len(out)                                                  |
        if np.array_equal(out_strides, in_strides) and np.array_equal(       |
            out_shape, in_shape                                              |
        ):                                                                   |
            for ordinal in prange(out_size):---------------------------------| #2
                out[ordinal] = fn(in_storage[ordinal])                       |
        else:                                                                |
            for ordinal in prange(out_size):---------------------------------| #3
                in_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)---------| #0
                out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)--------| #1
                to_index(ordinal, out_shape, out_index)                      |
                broadcast_index(out_index, out_shape, in_shape, in_index)    |
                position = index_to_position(in_index, in_strides)           |
                out_position = index_to_position(out_index, out_strides)     |
                out[out_position] = fn(float(in_storage[position]))          |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...

Fused loop summary:
+--0 has the following loops fused into it:
   +--1 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #2, #3, #0).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--3 is a parallel loop
   +--0 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (serial, fused with loop(s): 1)



Parallel region 0 (loop #3) had 1 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#3).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/users/rundong/github/mod3-rd1119/minitorch/fast_ops.py (192) is hoisted out of
the parallel loop labelled #3 (it will be performed before the loop is executed
and reused inside the loop):
   Allocation:: in_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/users/rundong/github/mod3-rd1119/minitorch/fast_ops.py (193) is hoisted out of
the parallel loop labelled #3 (it will be performed before the loop is executed
and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/users/rundong/github/mod3-rd1119/minitorch/fast_ops.py (226)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /users/rundong/github/mod3-rd1119/minitorch/fast_ops.py (226)
-----------------------------------------------------------------------------|loop #ID
    def _zip(                                                                |
        out: Storage,                                                        |
        out_shape: Shape,                                                    |
        out_strides: Strides,                                                |
        a_storage: Storage,                                                  |
        a_shape: Shape,                                                      |
        a_strides: Strides,                                                  |
        b_storage: Storage,                                                  |
        b_shape: Shape,                                                      |
        b_strides: Strides,                                                  |
    ) -> None:                                                               |
        # TODO: Implement for Task 3.1.                                      |
        out_size = len(out)                                                  |
        if (                                                                 |
            np.array_equal(out_shape, b_shape)                               |
            and np.array_equal(out_shape, a_shape)                           |
            and np.array_equal(out_strides, a_strides)                       |
            and np.array_equal(out_strides, b_strides)                       |
        ):                                                                   |
            for ordinal in prange(out_size):---------------------------------| #4
                out[ordinal] = fn(a_storage[ordinal], b_storage[ordinal])    |
        else:                                                                |
            for ordinal in prange(out_size):---------------------------------| #5
                out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)        |
                a_index: Index = np.empty(MAX_DIMS, dtype=np.int32)          |
                b_index: Index = np.empty(MAX_DIMS, dtype=np.int32)          |
                to_index(ordinal, out_shape, out_index)                      |
                broadcast_index(out_index, out_shape, a_shape, a_index)      |
                a_data = a_storage[index_to_position(a_index, a_strides)]    |
                broadcast_index(out_index, out_shape, b_shape, b_index)      |
                b_data = b_storage[index_to_position(b_index, b_strides)]    |
                out[index_to_position(out_index, out_strides)] = fn(         |
                    float(a_data), float(b_data)                             |
                )                                                            |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #4, #5).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/users/rundong/github/mod3-rd1119/minitorch/fast_ops.py (249) is hoisted out of
the parallel loop labelled #5 (it will be performed before the loop is executed
and reused inside the loop):
   Allocation:: out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/users/rundong/github/mod3-rd1119/minitorch/fast_ops.py (250) is hoisted out of
the parallel loop labelled #5 (it will be performed before the loop is executed
and reused inside the loop):
   Allocation:: a_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/users/rundong/github/mod3-rd1119/minitorch/fast_ops.py (251) is hoisted out of
the parallel loop labelled #5 (it will be performed before the loop is executed
and reused inside the loop):
   Allocation:: b_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/users/rundong/github/mod3-rd1119/minitorch/fast_ops.py (285)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /users/rundong/github/mod3-rd1119/minitorch/fast_ops.py (285)
--------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                |
        out: Storage,                                                           |
        out_shape: Shape,                                                       |
        out_strides: Strides,                                                   |
        a_storage: Storage,                                                     |
        a_shape: Shape,                                                         |
        a_strides: Strides,                                                     |
        reduce_dim: int,                                                        |
    ) -> None:                                                                  |
        # TODO: Implement for Task 3.1.                                         |
                                                                                |
        out_size: int = len(out)                                                |
        reduce_size: int = a_shape[reduce_dim]                                  |
        for ordinal in prange(out_size):----------------------------------------| #7
            out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)---------------| #6
            to_index(ordinal, out_shape, out_index)                             |
            a_ordinal = index_to_position(out_index, a_strides)                 |
            reduced_val = out[ordinal]                                          |
            for j in range(reduce_size):                                        |
                reduced_val = fn(                                               |
                    reduced_val,                                                |
                    float(a_storage[a_ordinal + j * a_strides[reduce_dim]]),    |
                )                                                               |
            out[ordinal] = reduced_val                                          |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #7, #6).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--7 is a parallel loop
   +--6 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--7 (parallel)
   +--6 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--7 (parallel)
   +--6 (serial)



Parallel region 0 (loop #7) had 0 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#7).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/users/rundong/github/mod3-rd1119/minitorch/fast_ops.py (299) is hoisted out of
the parallel loop labelled #7 (it will be performed before the loop is executed
and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/users/rundong/github/mod3-rd1119/minitorch/fast_ops.py (313)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /users/rundong/github/mod3-rd1119/minitorch/fast_ops.py (313)
---------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                           |
    out: Storage,                                                                      |
    out_shape: Shape,                                                                  |
    out_strides: Strides,                                                              |
    a_storage: Storage,                                                                |
    a_shape: Shape,                                                                    |
    a_strides: Strides,                                                                |
    b_storage: Storage,                                                                |
    b_shape: Shape,                                                                    |
    b_strides: Strides,                                                                |
) -> None:                                                                             |
    """NUMBA tensor matrix multiply function.                                          |
                                                                                       |
    Should work for any tensor shapes that broadcast as long as                        |
                                                                                       |
    ```                                                                                |
    assert a_shape[-1] == b_shape[-2]                                                  |
    ```                                                                                |
                                                                                       |
    Optimizations:                                                                     |
                                                                                       |
    * Outer loop in parallel                                                           |
    * No index buffers or function calls                                               |
    * Inner loop should have no global writes, 1 multiply.                             |
                                                                                       |
                                                                                       |
    Args:                                                                              |
    ----                                                                               |
        out (Storage): storage for `out` tensor                                        |
        out_shape (Shape): shape for `out` tensor                                      |
        out_strides (Strides): strides for `out` tensor                                |
        a_storage (Storage): storage for `a` tensor                                    |
        a_shape (Shape): shape for `a` tensor                                          |
        a_strides (Strides): strides for `a` tensor                                    |
        b_storage (Storage): storage for `b` tensor                                    |
        b_shape (Shape): shape for `b` tensor                                          |
        b_strides (Strides): strides for `b` tensor                                    |
                                                                                       |
    Returns:                                                                           |
    -------                                                                            |
        None : Fills in `out`                                                          |
                                                                                       |
    """                                                                                |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                             |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                             |
                                                                                       |
    # TODO: Implement for Task 3.2.                                                    |
    N = a_shape[-1]                                                                    |
    I, J, K = out_shape[-3:]                                                           |
    for i in prange(I):----------------------------------------------------------------| #10
        for j in prange(J):------------------------------------------------------------| #9
            for k in prange(K):--------------------------------------------------------| #8
                val = 0.0                                                              |
                a_ordinal = a_batch_stride * i + a_strides[-2] * j                     |
                b_ordinal = b_batch_stride * i + b_strides[-1] * k                     |
                for _ in range(N):                                                     |
                    val += a_storage[a_ordinal] * b_storage[b_ordinal]                 |
                    a_ordinal += a_strides[-1]                                         |
                    b_ordinal += b_strides[-2]                                         |
                out_ordinal = (                                                        |
                    out_strides[-1] * k + out_strides[-2] * j + out_strides[-3] * i    |
                )                                                                      |
                out[out_ordinal] = val                                                 |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #10, #9).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--10 is a parallel loop
   +--9 --> rewritten as a serial loop
      +--8 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (parallel)
      +--8 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (serial)
      +--8 (serial)



Parallel region 0 (loop #10) had 0 loop(s) fused and 2 loop(s) serialized as
part of the larger parallel loop (#10).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```
# Task 4

| Matrix Size          | FastOps Time (s) | CudaOps Time (s) |
|----------------------|------------------|------------------|
| 64                   | 0.00342         | 0.00641         |
| 128                  | 0.01582         | 0.01489         |
| 256                  | 0.09494         | 0.05744         |
| 512                  | 1.07415         | 0.21905         |
| 1024                 | 7.84241         | 0.99863         |
The graph for the comparation between Cuda Implementation and naive operations.
![task4-result](./Graph/Task4.png)

# Task5

## GPU, Hidden=100, Dataset=Split, Rate=0.05
   ```bash
   !cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
   ```

   ```bash
        Epoch  0  loss  7.030329622322549 correct 29 Time per epoch 0.38790576457977294
        Epoch  10  loss  4.496681386710301 correct 26 Time per epoch 1.6367682933807373
        Epoch  20  loss  6.128431978335912 correct 36 Time per epoch 1.5687076330184937
        Epoch  30  loss  4.370202298858919 correct 46 Time per epoch 1.561046314239502
        Epoch  40  loss  3.6281915183647877 correct 46 Time per epoch 1.576847457885742
        Epoch  50  loss  3.1357959770227106 correct 45 Time per epoch 1.6455533742904662
        Epoch  60  loss  2.162244334070718 correct 47 Time per epoch 1.5567980766296388
        Epoch  70  loss  2.6602194450890453 correct 45 Time per epoch 1.5652177810668946
        Epoch  80  loss  2.7124660858865672 correct 43 Time per epoch 1.7340365171432495
        Epoch  90  loss  1.6830174686967776 correct 48 Time per epoch 1.5639925956726075
        Epoch  100  loss  2.4295431377832513 correct 43 Time per epoch 1.562187623977661
        Epoch  110  loss  1.6566031100281162 correct 47 Time per epoch 1.5683364391326904
        Epoch  120  loss  2.1416442367919024 correct 47 Time per epoch 1.6363176822662353
        Epoch  130  loss  1.243159113703022 correct 48 Time per epoch 1.5588584184646606
        Epoch  140  loss  1.9213199064366395 correct 45 Time per epoch 1.5601646661758424
        Epoch  150  loss  2.2682643219923673 correct 48 Time per epoch 1.6348508596420288
        Epoch  160  loss  1.5838061958529246 correct 50 Time per epoch 1.5631211996078491
        Epoch  170  loss  1.8281198092424555 correct 49 Time per epoch 1.5629527807235717
        Epoch  180  loss  1.3376825719361338 correct 50 Time per epoch 1.5753697872161865
        Epoch  190  loss  1.1465268072414005 correct 46 Time per epoch 1.6182536602020263
        Epoch  200  loss  1.006569316897611 correct 48 Time per epoch 1.5613746166229248
        Epoch  210  loss  3.4060009951335757 correct 47 Time per epoch 1.5633638858795167
        Epoch  220  loss  4.264992780290969 correct 48 Time per epoch 1.6390291452407837
        Epoch  230  loss  0.8531605009705876 correct 50 Time per epoch 1.5603173494338989
        Epoch  240  loss  6.130987642861223 correct 38 Time per epoch 1.5764514684677124
        Epoch  250  loss  2.5826707630260684 correct 47 Time per epoch 1.5795851230621338
        Epoch  260  loss  1.1080700590362655 correct 48 Time per epoch 1.622667956352234
        Epoch  270  loss  0.7314500832138859 correct 45 Time per epoch 1.6566795587539673
        Epoch  280  loss  1.305768182815533 correct 47 Time per epoch 1.550727367401123
        Epoch  290  loss  1.0535672113660866 correct 49 Time per epoch 1.6345623016357422
        Epoch  300  loss  1.7207453308019889 correct 50 Time per epoch 1.5590502500534058
        Epoch  310  loss  0.5018154470962541 correct 45 Time per epoch 1.5623395919799805
        Epoch  320  loss  1.179527347257625 correct 49 Time per epoch 1.5539769649505615
        Epoch  330  loss  0.8173735946007702 correct 47 Time per epoch 1.6320308923721314
        Epoch  340  loss  2.0692489647649293 correct 48 Time per epoch 1.5556090593338012
        Epoch  350  loss  1.831574086915719 correct 50 Time per epoch 1.5602456092834474
        Epoch  360  loss  0.8058495521517599 correct 50 Time per epoch 1.6063069820404052
        Epoch  370  loss  1.126596935163321 correct 47 Time per epoch 1.5792619943618775
        Epoch  380  loss  1.5734018531253877 correct 47 Time per epoch 1.5524924516677856
        Epoch  390  loss  1.3064163183416955 correct 49 Time per epoch 1.56344735622406
        Epoch  400  loss  0.8470412570068735 correct 50 Time per epoch 1.6317063331604005
        Epoch  410  loss  0.37325089955354207 correct 50 Time per epoch 1.554289197921753
        Epoch  420  loss  0.794281589998707 correct 50 Time per epoch 1.5523608922958374
        Epoch  430  loss  0.1200985647721971 correct 47 Time per epoch 1.6176042795181274
        Epoch  440  loss  1.1198057735316644 correct 50 Time per epoch 1.5906382560729981
        Epoch  450  loss  0.45207294402710496 correct 50 Time per epoch 1.5524711608886719
        Epoch  460  loss  1.0801616462274302 correct 50 Time per epoch 1.5584780693054199
        Epoch  470  loss  0.8228507731021052 correct 49 Time per epoch 1.7180626392364502
        Epoch  480  loss  1.2685997379897098 correct 50 Time per epoch 1.5610052585601806
        Epoch  490  loss  1.3596379225545836 correct 50 Time per epoch 1.547406053543091
```
## Log for CPU, Hidden=100, Dataset=Split, Rate=0.05
```bash
   !cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
   ```

   ```bash
        Epoch  0  loss  6.641579905541806 correct 28 Time per epoch 1.435877799987793
        Epoch  10  loss  5.02394374441548 correct 39 Time per epoch 0.13884754180908204
        Epoch  20  loss  5.0456205238833505 correct 35 Time per epoch 0.20109705924987792
        Epoch  30  loss  3.7758766118571194 correct 45 Time per epoch 0.11016435623168945
        Epoch  40  loss  5.013064862517209 correct 47 Time per epoch 0.11224696636199952
        Epoch  50  loss  3.1928480765096827 correct 48 Time per epoch 0.11033351421356201
        Epoch  60  loss  1.898677206465932 correct 47 Time per epoch 0.11016056537628174
        Epoch  70  loss  1.6926557361662378 correct 48 Time per epoch 0.10958619117736816
        Epoch  80  loss  2.6076206100632975 correct 46 Time per epoch 0.111328387260437
        Epoch  90  loss  1.1044360522360688 correct 48 Time per epoch 0.10992507934570313
        Epoch  100  loss  0.5192699155888002 correct 47 Time per epoch 0.1095463514328003
        Epoch  110  loss  0.7293814221777412 correct 48 Time per epoch 0.11776885986328126
        Epoch  120  loss  1.4604518146580094 correct 48 Time per epoch 0.2018970251083374
        Epoch  130  loss  1.0012174708939476 correct 49 Time per epoch 0.12511012554168702
        Epoch  140  loss  0.46013293933984245 correct 49 Time per epoch 0.11060826778411866
        Epoch  150  loss  1.5640321819505851 correct 49 Time per epoch 0.10989663600921631
        Epoch  160  loss  0.4776525076228509 correct 48 Time per epoch 0.11024856567382812
        Epoch  170  loss  0.5219268148694662 correct 50 Time per epoch 0.10896446704864501
        Epoch  180  loss  1.9039126853449382 correct 48 Time per epoch 0.10982873439788818
        Epoch  190  loss  2.651305045391856 correct 46 Time per epoch 0.11057426929473876
        Epoch  200  loss  0.21443530289773122 correct 47 Time per epoch 0.11156687736511231
        Epoch  210  loss  0.8028541093197802 correct 49 Time per epoch 0.11119260787963867
        Epoch  220  loss  0.247567803453427 correct 49 Time per epoch 0.1933891534805298
        Epoch  230  loss  0.9880156754923732 correct 49 Time per epoch 0.14403142929077148
        Epoch  240  loss  0.5045651004385868 correct 48 Time per epoch 0.11205062866210938
        Epoch  250  loss  0.36031219207861975 correct 49 Time per epoch 0.10969560146331787
        Epoch  260  loss  0.4058905151236129 correct 49 Time per epoch 0.12019131183624268
        Epoch  270  loss  0.6611211506402405 correct 50 Time per epoch 0.11002864837646484
        Epoch  280  loss  0.8141287485185503 correct 49 Time per epoch 0.10982601642608643
        Epoch  290  loss  0.37206560607705824 correct 47 Time per epoch 0.11095519065856933
        Epoch  300  loss  0.31264022949534387 correct 50 Time per epoch 0.10984270572662354
        Epoch  310  loss  2.3265267659102475 correct 47 Time per epoch 0.1095996618270874
        Epoch  320  loss  3.3183812978696006 correct 48 Time per epoch 0.18856408596038818
        Epoch  330  loss  0.5566749662574594 correct 49 Time per epoch 0.14541981220245362
        Epoch  340  loss  0.6451899287347702 correct 49 Time per epoch 0.10907857418060303
        Epoch  350  loss  0.7620634570908436 correct 49 Time per epoch 0.1096238374710083
        Epoch  360  loss  1.0949108440881876 correct 48 Time per epoch 0.10984199047088623
        Epoch  370  loss  1.164550791590681 correct 50 Time per epoch 0.1108816385269165
        Epoch  380  loss  0.8764573431935663 correct 49 Time per epoch 0.10905251502990723
        Epoch  390  loss  0.16546967803650747 correct 49 Time per epoch 0.10902657508850097
        Epoch  400  loss  1.135167294115634 correct 50 Time per epoch 0.10967147350311279
        Epoch  410  loss  0.42901456007970606 correct 50 Time per epoch 0.10937027931213379
        Epoch  420  loss  0.04380896140439232 correct 49 Time per epoch 0.16070556640625
        Epoch  430  loss  0.14562949286234814 correct 49 Time per epoch 0.17720882892608641
        Epoch  440  loss  0.3200063404653361 correct 50 Time per epoch 0.10823979377746581
        Epoch  450  loss  0.2903542998082129 correct 50 Time per epoch 0.10912513732910156
        Epoch  460  loss  0.7944141668323299 correct 50 Time per epoch 0.11001882553100586
        Epoch  470  loss  1.0851101437286794 correct 49 Time per epoch 0.10805928707122803
        Epoch  480  loss  0.3666256340905247 correct 50 Time per epoch 0.10782449245452881
        Epoch  490  loss  0.24195862283988362 correct 49 Time per epoch 0.11262617111206055
   ```
## Log for GPU, Hidden=100, Dataset=Simple, Rate=0.05
```bash
   !cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05
   ```

   ```bash
        Epoch  0  loss  4.231057618503737 correct 40 Time per epoch 0.4488811016082764
        Epoch  10  loss  0.8467802908955622 correct 49 Time per epoch 1.5690074920654298
        Epoch  20  loss  1.6482978034940654 correct 48 Time per epoch 1.5706833124160766
        Epoch  30  loss  0.6324838504066655 correct 48 Time per epoch 1.642581057548523
        Epoch  40  loss  0.8950996673174823 correct 49 Time per epoch 1.5805742502212525
        Epoch  50  loss  0.557809176628955 correct 48 Time per epoch 1.570874238014221
        Epoch  60  loss  1.3102887662350673 correct 50 Time per epoch 1.6063580989837647
        Epoch  70  loss  1.697801879824845 correct 48 Time per epoch 1.6199917793273926
        Epoch  80  loss  0.08106413440341503 correct 48 Time per epoch 1.5719878435134889
        Epoch  90  loss  0.4067537828134001 correct 48 Time per epoch 1.5644217252731323
        Epoch  100  loss  0.0742427889841448 correct 50 Time per epoch 1.6503339767456056
        Epoch  110  loss  0.4756091691101141 correct 49 Time per epoch 1.6955562353134155
        Epoch  120  loss  1.3418731025636523 correct 49 Time per epoch 1.5934258222579956
        Epoch  130  loss  0.5380740192756103 correct 49 Time per epoch 1.6679628849029542
        Epoch  140  loss  0.012924391166384386 correct 49 Time per epoch 1.5774704456329345
        Epoch  150  loss  0.007138768457189043 correct 47 Time per epoch 1.5773017406463623
        Epoch  160  loss  0.1521845876640417 correct 50 Time per epoch 1.6427552938461303
        Epoch  170  loss  0.06101198209403101 correct 50 Time per epoch 1.576206588745117
        Epoch  180  loss  0.36331552958001045 correct 49 Time per epoch 1.5633602857589721
        Epoch  190  loss  0.6858176067364078 correct 50 Time per epoch 1.5948326349258424
        Epoch  200  loss  0.8743589780061779 correct 48 Time per epoch 1.6196994304656982
        Epoch  210  loss  0.9137573520419653 correct 50 Time per epoch 1.5781194925308228
        Epoch  220  loss  0.12973097008031595 correct 50 Time per epoch 1.5659207344055175
        Epoch  230  loss  0.333473533519198 correct 49 Time per epoch 1.6490115165710448
        Epoch  240  loss  0.5801070481921121 correct 49 Time per epoch 1.5623791217803955
        Epoch  250  loss  0.44292662517255305 correct 49 Time per epoch 1.5688508749008179
        Epoch  260  loss  1.2806590539121183 correct 50 Time per epoch 1.6387576341629029
        Epoch  270  loss  0.7524215291097837 correct 50 Time per epoch 1.5839954614639282
        Epoch  280  loss  0.659093157008466 correct 50 Time per epoch 1.5667675018310547
        Epoch  290  loss  0.08920144568287817 correct 50 Time per epoch 1.5662221670150758
        Epoch  300  loss  1.6115410218755684 correct 48 Time per epoch 1.743427538871765
        Epoch  310  loss  0.86187292769938 correct 50 Time per epoch 1.5690655708312988
        Epoch  320  loss  0.5156947497816096 correct 49 Time per epoch 1.5733680963516234
        Epoch  330  loss  0.5976275455953565 correct 49 Time per epoch 1.6447987794876098
        Epoch  340  loss  0.11143267311434368 correct 49 Time per epoch 1.5705573797225951
        Epoch  350  loss  0.4709308173535459 correct 49 Time per epoch 1.5632790327072144
        Epoch  360  loss  1.7302280438197948 correct 48 Time per epoch 1.5826247930526733
        Epoch  370  loss  1.135732244371917 correct 50 Time per epoch 1.631567358970642
        Epoch  380  loss  0.7934008748515493 correct 49 Time per epoch 1.5881070852279664
        Epoch  390  loss  0.7718463037146869 correct 50 Time per epoch 1.5788071393966674
        Epoch  400  loss  1.6858438734811 correct 48 Time per epoch 1.660209321975708
        Epoch  410  loss  0.16169846891747558 correct 49 Time per epoch 1.5730446815490722
        Epoch  420  loss  1.1804958734110034 correct 50 Time per epoch 1.5614545345306396
        Epoch  430  loss  0.006436953273244205 correct 50 Time per epoch 1.618458914756775
        Epoch  440  loss  0.5896515584878574 correct 50 Time per epoch 1.5990609169006347
        Epoch  450  loss  0.23801012726541856 correct 50 Time per epoch 1.5805660724639892
        Epoch  460  loss  0.20424528272422537 correct 50 Time per epoch 1.5661282777786254
        Epoch  470  loss  0.5736815671557478 correct 49 Time per epoch 1.647271203994751
        Epoch  480  loss  0.025648743403433344 correct 49 Time per epoch 1.5685143232345582
        Epoch  490  loss  0.4852364551562155 correct 50 Time per epoch 1.566480803489685
```
## Log for CPU, Hidden=100, Dataset=Simple, Rate=0.05
```bash
   !cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05
   ```

   ```bash
        Epoch  0  loss  4.7964597120985655 correct 45 Time per epoch 1.4510001420974732
        Epoch  10  loss  1.843893697153623 correct 49 Time per epoch 0.18239452838897705
        Epoch  20  loss  1.1800711870149345 correct 47 Time per epoch 0.15107412338256837
        Epoch  30  loss  1.1379222704817513 correct 49 Time per epoch 0.11080653667449951
        Epoch  40  loss  1.5238688027902942 correct 50 Time per epoch 0.1107985258102417
        Epoch  50  loss  1.0257448933380902 correct 49 Time per epoch 0.11265122890472412
        Epoch  60  loss  1.3678295992416851 correct 50 Time per epoch 0.11178324222564698
        Epoch  70  loss  0.6147506460680406 correct 49 Time per epoch 0.11113805770874023
        Epoch  80  loss  1.330776877652118 correct 50 Time per epoch 0.11076166629791259
        Epoch  90  loss  0.30768964966024254 correct 50 Time per epoch 0.11039502620697021
        Epoch  100  loss  1.0228521731630231 correct 50 Time per epoch 0.11044423580169678
        Epoch  110  loss  0.22464174361273043 correct 50 Time per epoch 0.17132461071014404
        Epoch  120  loss  2.1040612290873737 correct 47 Time per epoch 0.17105562686920167
        Epoch  130  loss  0.5359279796223502 correct 50 Time per epoch 0.11143200397491455
        Epoch  140  loss  0.3934004666180992 correct 49 Time per epoch 0.11094987392425537
        Epoch  150  loss  0.9526635705044021 correct 49 Time per epoch 0.11008434295654297
        Epoch  160  loss  0.6443498056264325 correct 49 Time per epoch 0.11074340343475342
        Epoch  170  loss  0.9802280033170013 correct 50 Time per epoch 0.11117870807647705
        Epoch  180  loss  0.7189045145327252 correct 50 Time per epoch 0.10961849689483642
        Epoch  190  loss  1.5098887443018265 correct 50 Time per epoch 0.11025395393371581
        Epoch  200  loss  0.0014299828378507374 correct 50 Time per epoch 0.1103811502456665
        Epoch  210  loss  0.2593636990953903 correct 50 Time per epoch 0.14995067119598388
        Epoch  220  loss  0.8668796820361435 correct 50 Time per epoch 0.189056134223938
        Epoch  230  loss  0.20065615357066824 correct 49 Time per epoch 0.11216158866882324
        Epoch  240  loss  0.9751354778464866 correct 50 Time per epoch 0.10978918075561524
        Epoch  250  loss  0.051120279881150726 correct 50 Time per epoch 0.11131904125213624
        Epoch  260  loss  0.0297672604690793 correct 50 Time per epoch 0.11813881397247314
        Epoch  270  loss  1.0504312831647271 correct 49 Time per epoch 0.10984952449798584
        Epoch  280  loss  0.4809041622152252 correct 50 Time per epoch 0.10946667194366455
        Epoch  290  loss  0.19378936222149476 correct 50 Time per epoch 0.11000027656555175
        Epoch  300  loss  0.22538555097529747 correct 50 Time per epoch 0.11441235542297364
        Epoch  310  loss  1.308985161962895 correct 49 Time per epoch 0.1533799171447754
        Epoch  320  loss  0.0125820090077368 correct 50 Time per epoch 0.1890047788619995
        Epoch  330  loss  0.4738793137849225 correct 50 Time per epoch 0.10994317531585693
        Epoch  340  loss  0.33605274138434077 correct 50 Time per epoch 0.11056420803070069
        Epoch  350  loss  0.3870854307952717 correct 50 Time per epoch 0.10993781089782714
        Epoch  360  loss  1.0214881093035832 correct 49 Time per epoch 0.11386642456054688
        Epoch  370  loss  0.13447224321815573 correct 50 Time per epoch 0.1100649356842041
        Epoch  380  loss  0.8415821865544987 correct 50 Time per epoch 0.11104109287261962
        Epoch  390  loss  0.2991686170031502 correct 50 Time per epoch 0.1113133430480957
        Epoch  400  loss  0.025445227397062037 correct 49 Time per epoch 0.10999495983123779
        Epoch  410  loss  0.4636795495992929 correct 50 Time per epoch 0.1456385850906372
        Epoch  420  loss  0.28358573345547494 correct 50 Time per epoch 0.1933736801147461
        Epoch  430  loss  0.31946395312499154 correct 50 Time per epoch 0.10944614410400391
        Epoch  440  loss  0.08558653286112176 correct 50 Time per epoch 0.1090163230895996
        Epoch  450  loss  0.2065363939060571 correct 50 Time per epoch 0.108502197265625
        Epoch  460  loss  0.06460771672402452 correct 50 Time per epoch 0.11180846691131592
        Epoch  470  loss  0.5052649667243666 correct 50 Time per epoch 0.1108548402786255
        Epoch  480  loss  0.1963802396358422 correct 50 Time per epoch 0.11024737358093262
        Epoch  490  loss  0.28835663129626127 correct 50 Time per epoch 0.1095313310623169
```
## Log for GPU, Hidden=100, Dataset=Xor, Rate=0.05
```bash
   !cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05
   ```

   ```bash
        Epoch  0  loss  7.357198650204386 correct 38 Time per epoch 0.4482653856277466
        Epoch  10  loss  4.98915897964179 correct 45 Time per epoch 1.5824486494064331
        Epoch  20  loss  2.1976805743230505 correct 46 Time per epoch 1.5897902011871339
        Epoch  30  loss  2.9004835676412393 correct 45 Time per epoch 1.6372377634048463
        Epoch  40  loss  3.251250887754409 correct 47 Time per epoch 1.5820262908935547
        Epoch  50  loss  1.4796033739767882 correct 46 Time per epoch 1.5812280178070068
        Epoch  60  loss  4.639377154234314 correct 46 Time per epoch 1.6576990842819215
        Epoch  70  loss  2.078155600936099 correct 47 Time per epoch 1.5703636407852173
        Epoch  80  loss  1.5316967925777674 correct 48 Time per epoch 1.5811646938323975
        Epoch  90  loss  0.5446958954844076 correct 48 Time per epoch 1.6475631952285767
        Epoch  100  loss  2.449873778870412 correct 47 Time per epoch 1.590339970588684
        Epoch  110  loss  0.38874546992140546 correct 48 Time per epoch 1.5709888696670533
        Epoch  120  loss  1.7194360639369466 correct 49 Time per epoch 1.5945136070251464
        Epoch  130  loss  1.8249991870734452 correct 49 Time per epoch 1.636206293106079
        Epoch  140  loss  0.5861702383672712 correct 48 Time per epoch 1.6637208700180053
        Epoch  150  loss  0.3929063224952324 correct 49 Time per epoch 1.570684552192688
        Epoch  160  loss  2.1724459667772043 correct 48 Time per epoch 1.6559552669525146
        Epoch  170  loss  0.47417269001601015 correct 49 Time per epoch 1.579501748085022
        Epoch  180  loss  1.5813246868671698 correct 49 Time per epoch 1.5714139223098755
        Epoch  190  loss  1.2426094233540756 correct 49 Time per epoch 1.6513227462768554
        Epoch  200  loss  1.739378840943092 correct 49 Time per epoch 1.5774714469909668
        Epoch  210  loss  2.439928648114148 correct 49 Time per epoch 1.576958131790161
        Epoch  220  loss  0.603007004008531 correct 50 Time per epoch 1.5986249685287475
        Epoch  230  loss  1.1203620081816774 correct 49 Time per epoch 1.6287386178970338
        Epoch  240  loss  0.627594984812291 correct 49 Time per epoch 1.5701635837554933
        Epoch  250  loss  0.9752555552534153 correct 50 Time per epoch 1.5766926288604737
        Epoch  260  loss  0.31317071211636416 correct 50 Time per epoch 1.6579359769821167
        Epoch  270  loss  0.5065025871551336 correct 49 Time per epoch 1.5747862815856934
        Epoch  280  loss  0.7243142947917529 correct 50 Time per epoch 1.5748349666595458
        Epoch  290  loss  0.9050876533395467 correct 49 Time per epoch 1.6506150722503663
        Epoch  300  loss  0.519815417007563 correct 49 Time per epoch 1.592959475517273
        Epoch  310  loss  1.704718279594221 correct 50 Time per epoch 1.5740541458129882
        Epoch  320  loss  0.7543478790454056 correct 49 Time per epoch 1.5653656721115112
        Epoch  330  loss  0.19069303999732004 correct 50 Time per epoch 1.6585341691970825
        Epoch  340  loss  0.991318715116128 correct 50 Time per epoch 1.6590985298156737
        Epoch  350  loss  1.2631215955057693 correct 50 Time per epoch 1.5746696710586547
        Epoch  360  loss  0.6695773459732339 correct 49 Time per epoch 1.6562577486038208
        Epoch  370  loss  0.2527241426419322 correct 49 Time per epoch 1.5721393346786499
        Epoch  380  loss  0.43344854133060023 correct 49 Time per epoch 1.5791359901428224
        Epoch  390  loss  0.10489459731722195 correct 50 Time per epoch 1.650852632522583
        Epoch  400  loss  0.10007497312113778 correct 49 Time per epoch 1.5767351388931274
        Epoch  410  loss  1.3404430869320194 correct 50 Time per epoch 1.5691748142242432
        Epoch  420  loss  0.41423757348346174 correct 50 Time per epoch 1.5768578767776489
        Epoch  430  loss  0.6058951313524595 correct 50 Time per epoch 1.6487502336502076
        Epoch  440  loss  0.9870924226237183 correct 49 Time per epoch 1.5709683418273925
        Epoch  450  loss  0.03946025383127734 correct 49 Time per epoch 1.5713172435760498
        Epoch  460  loss  0.23711322278759045 correct 50 Time per epoch 1.6708690404891968
        Epoch  470  loss  0.101533323597164 correct 50 Time per epoch 1.5742628812789916
        Epoch  480  loss  0.8063855624185929 correct 50 Time per epoch 1.5875962972640991
        Epoch  490  loss  0.04363320153761432 correct 50 Time per epoch 1.6209076166152954
```
## Log for CPU, Hidden=100, Dataset=Xor, Rate=0.05
```bash
   !cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05
   ```

   ```bash
        Epoch  0  loss  6.528275023369353 correct 22 Time per epoch 1.4627237558364867
        Epoch  10  loss  4.2877176883964925 correct 47 Time per epoch 0.1110661506652832
        Epoch  20  loss  4.324374417950544 correct 41 Time per epoch 0.11166291236877442
        Epoch  30  loss  1.7118092404117968 correct 49 Time per epoch 0.11279699802398682
        Epoch  40  loss  2.050872327632735 correct 49 Time per epoch 0.11217312812805176
        Epoch  50  loss  1.597773911418098 correct 46 Time per epoch 0.11132428646087647
        Epoch  60  loss  1.9819400698609726 correct 49 Time per epoch 0.11056866645812988
        Epoch  70  loss  1.639872900228014 correct 49 Time per epoch 0.11049323081970215
        Epoch  80  loss  1.181799442978592 correct 49 Time per epoch 0.11202948093414307
        Epoch  90  loss  1.8767653337925143 correct 49 Time per epoch 0.17596683502197266
        Epoch  100  loss  0.8371002718860567 correct 49 Time per epoch 0.16111915111541747
        Epoch  110  loss  0.39023110316073695 correct 50 Time per epoch 0.11271145343780517
        Epoch  120  loss  0.45715818298824334 correct 49 Time per epoch 0.11169359683990479
        Epoch  130  loss  1.2340055267279115 correct 50 Time per epoch 0.1110102653503418
        Epoch  140  loss  0.6103370302286064 correct 50 Time per epoch 0.11323065757751465
        Epoch  150  loss  1.0901422392690718 correct 50 Time per epoch 0.1114161491394043
        Epoch  160  loss  0.12445051759086685 correct 49 Time per epoch 0.11085991859436035
        Epoch  170  loss  0.31381787922736165 correct 50 Time per epoch 0.1123319149017334
        Epoch  180  loss  0.18691340994488548 correct 50 Time per epoch 0.19444496631622316
        Epoch  190  loss  0.2447200281343434 correct 50 Time per epoch 0.24259014129638673
        Epoch  200  loss  0.8053217984854636 correct 50 Time per epoch 0.13876669406890868
        Epoch  210  loss  0.3099817587087687 correct 50 Time per epoch 0.11080949306488037
        Epoch  220  loss  1.4113461682469302 correct 50 Time per epoch 0.11123373508453369
        Epoch  230  loss  0.969516500547963 correct 50 Time per epoch 0.11069614887237549
        Epoch  240  loss  0.06718547249129793 correct 49 Time per epoch 0.11010541915893554
        Epoch  250  loss  0.3744387949657912 correct 50 Time per epoch 0.11114580631256103
        Epoch  260  loss  0.5080310573217296 correct 50 Time per epoch 0.11961197853088379
        Epoch  270  loss  1.1513708577501705 correct 50 Time per epoch 0.11292452812194824
        Epoch  280  loss  0.2699568395408089 correct 50 Time per epoch 0.11033556461334229
        Epoch  290  loss  0.23776245380586208 correct 50 Time per epoch 0.2011582851409912
        Epoch  300  loss  0.0746719955093508 correct 49 Time per epoch 0.13878445625305175
        Epoch  310  loss  0.5396445128543039 correct 50 Time per epoch 0.11117193698883057
        Epoch  320  loss  0.8533089801582873 correct 50 Time per epoch 0.11038179397583008
        Epoch  330  loss  1.024858339924342 correct 50 Time per epoch 0.11232957839965821
        Epoch  340  loss  0.18230641260539254 correct 50 Time per epoch 0.1115645170211792
        Epoch  350  loss  0.5712081763650627 correct 50 Time per epoch 0.11272530555725098
        Epoch  360  loss  0.6360743500983793 correct 50 Time per epoch 0.11066064834594727
        Epoch  370  loss  0.16007146144615675 correct 50 Time per epoch 0.10993049144744874
        Epoch  380  loss  0.22985502798455046 correct 50 Time per epoch 0.10964956283569335
        Epoch  390  loss  0.33808866647660585 correct 50 Time per epoch 0.18424456119537352
        Epoch  400  loss  0.19664080338214357 correct 50 Time per epoch 0.15303208827972412
        Epoch  410  loss  0.5527057703255501 correct 50 Time per epoch 0.10928518772125244
        Epoch  420  loss  0.02455244434635427 correct 50 Time per epoch 0.1106644868850708
        Epoch  430  loss  0.4495345732781258 correct 50 Time per epoch 0.11852936744689942
        Epoch  440  loss  0.45632758259547196 correct 50 Time per epoch 0.11136507987976074
        Epoch  450  loss  0.10386197237110838 correct 50 Time per epoch 0.11001877784729004
        Epoch  460  loss  0.4457662423588716 correct 50 Time per epoch 0.11058964729309081
        Epoch  470  loss  0.08569770107300596 correct 50 Time per epoch 0.11004478931427002
        Epoch  480  loss  0.30173596987494317 correct 50 Time per epoch 0.11073358058929443
        Epoch  490  loss  0.1881854680537514 correct 50 Time per epoch 0.1792076826095581
```
## Log for GPU, Hidden=200, Dataset=Simple, Rate=0.05
```bash
   !cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 200 --DATASET simple --RATE 0.055
   ```

   ```bash
        Epoch  0  loss  7.000360093429833 correct 43 Time per epoch 0.3491117000579834
        Epoch  10  loss  3.775766526274423 correct 43 Time per epoch 1.6753533601760864
        Epoch  20  loss  0.4885228371957832 correct 45 Time per epoch 1.729252004623413
        Epoch  30  loss  0.4872548281381251 correct 50 Time per epoch 1.6713150024414063
        Epoch  40  loss  0.9365585218427813 correct 50 Time per epoch 1.757141351699829
        Epoch  50  loss  0.8800666177591823 correct 50 Time per epoch 1.658739948272705
        Epoch  60  loss  1.2337028794483187 correct 50 Time per epoch 1.6615067005157471
        Epoch  70  loss  0.9998951260407065 correct 50 Time per epoch 1.739526915550232
        Epoch  80  loss  0.6160068508465498 correct 50 Time per epoch 1.6656420946121215
        Epoch  90  loss  0.14959700058122263 correct 48 Time per epoch 1.6556017637252807
        Epoch  100  loss  0.9650618103263872 correct 49 Time per epoch 1.7389044523239137
        Epoch  110  loss  1.0009910986218675 correct 50 Time per epoch 1.6674356698989867
        Epoch  120  loss  0.001862541799106978 correct 46 Time per epoch 1.7281122922897338
        Epoch  130  loss  0.6105513975878184 correct 47 Time per epoch 1.6667046785354613
        Epoch  140  loss  1.2342749221129032 correct 48 Time per epoch 1.6610356330871583
        Epoch  150  loss  0.3115650682857444 correct 49 Time per epoch 1.7372191905975343
        Epoch  160  loss  1.0192190028056882 correct 50 Time per epoch 1.7474852323532104
        Epoch  170  loss  0.7488148680971483 correct 50 Time per epoch 1.6661609172821046
        Epoch  180  loss  0.34162930515686135 correct 50 Time per epoch 1.73691885471344
        Epoch  190  loss  0.5085624737347073 correct 50 Time per epoch 1.654402995109558
        Epoch  200  loss  0.35635743233124584 correct 50 Time per epoch 1.6527538299560547
        Epoch  210  loss  0.0027412447387808723 correct 49 Time per epoch 1.7380970478057862
        Epoch  220  loss  0.5110860491183777 correct 50 Time per epoch 1.648557996749878
        Epoch  230  loss  0.6135522204946149 correct 50 Time per epoch 1.7002411842346192
        Epoch  240  loss  0.41265953848899467 correct 50 Time per epoch 1.6890200138092042
        Epoch  250  loss  0.727802792051781 correct 50 Time per epoch 1.6535060405731201
        Epoch  260  loss  0.291619919556552 correct 50 Time per epoch 1.7456716060638429
        Epoch  270  loss  0.9377083411695825 correct 49 Time per epoch 1.6549450397491454
        Epoch  280  loss  0.07503656665791832 correct 50 Time per epoch 1.653823971748352
        Epoch  290  loss  0.606209333043264 correct 50 Time per epoch 1.7398022890090943
        Epoch  300  loss  0.005712076748624993 correct 50 Time per epoch 1.6587799787521362
        Epoch  310  loss  1.305381885987269 correct 50 Time per epoch 1.663024640083313
        Epoch  320  loss  7.262454332217169e-05 correct 50 Time per epoch 1.7148625373840332
        Epoch  330  loss  0.2859040339943481 correct 50 Time per epoch 1.6576553106307983
        Epoch  340  loss  0.44132882067538715 correct 50 Time per epoch 1.7245872259140014
        Epoch  350  loss  0.000583242454226611 correct 50 Time per epoch 1.746854591369629
        Epoch  360  loss  0.22734204775246558 correct 50 Time per epoch 1.645559597015381
        Epoch  370  loss  0.5344641971716934 correct 49 Time per epoch 1.7346656560897826
        Epoch  380  loss  0.4915285364830179 correct 50 Time per epoch 1.655346941947937
        Epoch  390  loss  6.512264483145665e-05 correct 50 Time per epoch 1.674775528907776
        Epoch  400  loss  0.003437713065598624 correct 50 Time per epoch 1.7150470972061158
        Epoch  410  loss  0.21421540051874122 correct 50 Time per epoch 1.6459196567535401
        Epoch  420  loss  0.2587896740875578 correct 50 Time per epoch 1.723518395423889
        Epoch  430  loss  0.001360230277267349 correct 47 Time per epoch 1.6692610263824463
        Epoch  440  loss  0.06037201859554331 correct 50 Time per epoch 1.6503085374832154
        Epoch  450  loss  0.0007078390972143175 correct 50 Time per epoch 1.7316558837890625
        Epoch  460  loss  0.07356948128335573 correct 50 Time per epoch 1.6542655944824218
        Epoch  470  loss  1.1509354033448206e-05 correct 50 Time per epoch 1.664808988571167
        Epoch  480  loss  0.31382458454050094 correct 50 Time per epoch 1.7450776100158691
        Epoch  490  loss  1.0406085233200262 correct 50 Time per epoch 1.6513777494430542
```
## Log for CPU, Hidden=200, Dataset=Simple, Rate=0.05
```bash
   !cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 200 --DATASET simple --RATE 0.05
   ```

   ```bash
        Epoch  0  loss  2.0323814824257833 correct 46 Time per epoch 1.4579288244247437
        Epoch  10  loss  0.34575115264416956 correct 49 Time per epoch 0.3570537567138672
        Epoch  20  loss  1.102252699682478 correct 49 Time per epoch 0.2724740982055664
        Epoch  30  loss  0.443838788253701 correct 50 Time per epoch 0.26042296886444094
        Epoch  40  loss  0.10342929093624108 correct 50 Time per epoch 0.2583003520965576
        Epoch  50  loss  0.9342391505405047 correct 50 Time per epoch 0.27146153450012206
        Epoch  60  loss  1.471104926292973 correct 50 Time per epoch 0.3571170330047607
        Epoch  70  loss  0.3598319667980328 correct 50 Time per epoch 0.2631401777267456
        Epoch  80  loss  0.1469966358469223 correct 50 Time per epoch 0.37769064903259275
        Epoch  90  loss  0.37804712288405734 correct 50 Time per epoch 0.3115377902984619
        Epoch  100  loss  0.03790966254360692 correct 50 Time per epoch 0.3153393268585205
        Epoch  110  loss  0.8547582488087573 correct 50 Time per epoch 0.25612964630126955
        Epoch  120  loss  0.08934128960002835 correct 50 Time per epoch 0.258467698097229
        Epoch  130  loss  0.06821811041644725 correct 50 Time per epoch 0.2587639570236206
        Epoch  140  loss  0.04257361647084147 correct 50 Time per epoch 0.37181222438812256
        Epoch  150  loss  0.5684944903933333 correct 50 Time per epoch 0.25813376903533936
        Epoch  160  loss  0.3845847617564051 correct 50 Time per epoch 0.2569802522659302
        Epoch  170  loss  0.017972094190366234 correct 50 Time per epoch 0.2622771978378296
        Epoch  180  loss  0.35939495067711097 correct 50 Time per epoch 0.37165882587432864
        Epoch  190  loss  0.5554534043878767 correct 50 Time per epoch 0.2573284864425659
        Epoch  200  loss  0.12310127484553898 correct 50 Time per epoch 0.2555901050567627
        Epoch  210  loss  0.0398237935775945 correct 50 Time per epoch 0.2555456876754761
        Epoch  220  loss  0.010914764197801708 correct 50 Time per epoch 0.29503955841064455
        Epoch  230  loss  0.03945617353655968 correct 50 Time per epoch 0.3365968942642212
        Epoch  240  loss  0.08271958762773494 correct 50 Time per epoch 0.2576359510421753
        Epoch  250  loss  0.34690382100577793 correct 50 Time per epoch 0.2565868616104126
        Epoch  260  loss  0.011445607681785172 correct 50 Time per epoch 0.2640532493591309
        Epoch  270  loss  0.3506395617528238 correct 50 Time per epoch 0.36884050369262694
        Epoch  280  loss  0.08424174108383144 correct 50 Time per epoch 0.26000959873199464
        Epoch  290  loss  0.002447075990270569 correct 50 Time per epoch 0.2545535802841187
        Epoch  300  loss  0.0076666266775685665 correct 50 Time per epoch 0.2555701732635498
        Epoch  310  loss  0.32062409828833444 correct 50 Time per epoch 0.3596575975418091
        Epoch  320  loss  0.09448048493864822 correct 50 Time per epoch 0.2650566577911377
        Epoch  330  loss  0.4060436270288088 correct 50 Time per epoch 0.2548715591430664
        Epoch  340  loss  0.2770055559264609 correct 50 Time per epoch 0.255192494392395
        Epoch  350  loss  0.3753169086878286 correct 50 Time per epoch 0.2688390493392944
        Epoch  360  loss  0.07837431100481666 correct 50 Time per epoch 0.3532301664352417
        Epoch  370  loss  0.010608165536730843 correct 50 Time per epoch 0.2562769651412964
        Epoch  380  loss  0.26798849908942096 correct 50 Time per epoch 0.2591264247894287
        Epoch  390  loss  0.2784278162165263 correct 50 Time per epoch 0.25400404930114745
        Epoch  400  loss  0.0586052932419302 correct 50 Time per epoch 0.37382590770721436
        Epoch  410  loss  0.0008723194020494614 correct 50 Time per epoch 0.2584576368331909
        Epoch  420  loss  0.2586216320384123 correct 50 Time per epoch 0.25503220558166506
        Epoch  430  loss  0.06051031530336377 correct 50 Time per epoch 0.2555375576019287
        Epoch  440  loss  0.19242953567494075 correct 50 Time per epoch 0.32076098918914797
        Epoch  450  loss  2.4822851854637323e-05 correct 50 Time per epoch 0.3047224521636963
        Epoch  460  loss  0.010551584028711881 correct 50 Time per epoch 0.2558619499206543
        Epoch  470  loss  0.3453719328026343 correct 50 Time per epoch 0.2549257755279541
        Epoch  480  loss  0.06445386826421769 correct 50 Time per epoch 0.25376651287078855
        Epoch  490  loss  0.06494521617249414 correct 50 Time per epoch 0.3757734775543213
```