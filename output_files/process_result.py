import re

with open("ppp.txt", "r") as f:
    result = f.readlines()

result = [i.split(" ") for i in result]


Last login: Thu May 30 23:13:23 on console
cd '/Users/nyxfer/Docu/PJ_GCN/';clear
(base) Haos-MacBook-Pro:~ nyxfer$ cd '/Users/nyxfer/Docu/PJ_GCN/';clear
(base) Haos-MacBook-Pro:PJ_GCN nyxfer$ source activate gcn
(gcn) Haos-MacBook-Pro:PJ_GCN nyxfer$ python main.py
The training edge types are:  [130, 644, 668, 1113, 1246, 762, 130, 644, 668, 1113, 1246, 762]
loading...
load gene_gene finished!
load gene_drug finished!
load drug_drug finished!
load drug_feature finished!
Edge types: 16
======================================================
Defining placeholders
Create minibatch iterator
Minibatch edge type: (1, 1, 0)
Train edges= 0480
Val edges= 0060
Test edges= 0060
Minibatch edge type: (1, 1, 1)
Train edges= 0409
Val edges= 0050
Test edges= 0050
Minibatch edge type: (1, 1, 2)
Train edges= 0408
Val edges= 0050
Test edges= 0050
Minibatch edge type: (1, 1, 3)
Train edges= 0598
Val edges= 0074
Test edges= 0074
Minibatch edge type: (1, 1, 4)
Train edges= 0443
Val edges= 0055
Test edges= 0055
Minibatch edge type: (1, 1, 5)
Train edges= 0731
Val edges= 0091
Test edges= 0091
Minibatch edge type: (1, 1, 6)
Train edges= 0480
Val edges= 0060
Test edges= 0060
Minibatch edge type: (1, 1, 7)
Train edges= 0409
Val edges= 0050
Test edges= 0050
Minibatch edge type: (1, 1, 8)
Train edges= 0408
Val edges= 0050
Test edges= 0050
Minibatch edge type: (1, 1, 9)
Train edges= 0598
Val edges= 0074
Test edges= 0074
Minibatch edge type: (1, 1, 10)
Train edges= 0443
Val edges= 0055
Test edges= 0055
Minibatch edge type: (1, 1, 11)
Train edges= 0731
Val edges= 0091
Test edges= 0091
mini-batch created!
Mini-batch finished!

Create model
WARNING:tensorflow:From /Users/nyxfer/Docu/PJ_GCN/decagon/deep/layers.py:93: calling l2_normalize (from tensorflow.python.ops.nn_impl) with dim is deprecated and will be removed in a future version.
Instructions for updating:
dim is deprecated, use axis instead
Create optimizer
WARNING:tensorflow:tf.op_scope(values, name, default_name) is deprecated, use tf.name_scope(name, default_name, values)
WARNING:tensorflow:tf.op_scope(values, name, default_name) is deprecated, use tf.name_scope(name, default_name, values)
/Users/nyxfer/anaconda3/envs/gcn/lib/python3.6/site-packages/tensorflow/python/ops/gradients_impl.py:100: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
    "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
Initialize session
2019-05-30 23:44:05.811538: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
Preparation Time Cost: 4.387042
Train model
Epoch: 0001 Iter: 0001 Edge: 0000 train_loss= 79.07414 val_roc= 0.72012 val_auprc= 0.71840 val_apk= 0.77864 time= 11.70928
Epoch: 0001 Iter: 0151 Edge: 0003 train_loss= 1.58478 val_roc= 0.90525 val_auprc= 0.91540 val_apk= 0.86237 time= 0.51900
Epoch: 0001 Iter: 0301 Edge: 0000 train_loss= 13.93918 val_roc= 0.83651 val_auprc= 0.82397 val_apk= 0.88832 time= 11.33933
Epoch: 0001 Iter: 0451 Edge: 0003 train_loss= 1.80169 val_roc= 0.89128 val_auprc= 0.89564 val_apk= 0.80127 time= 0.58610
Epoch: 0001 Iter: 0601 Edge: 0000 train_loss= 12.60385 val_roc= 0.85880 val_auprc= 0.84051 val_apk= 0.97559 time= 11.03255
Epoch: 0001 Iter: 0751 Edge: 0003 train_loss= 1.47094 val_roc= 0.84894 val_auprc= 0.84219 val_apk= 0.84478 time= 0.51590
Epoch: 0001 Iter: 0901 Edge: 0000 train_loss= 11.63447 val_roc= 0.87515 val_auprc= 0.85947 val_apk= 0.88964 time= 10.81737
Epoch: 0001 Iter: 1051 Edge: 0003 train_loss= 0.94812 val_roc= 0.86047 val_auprc= 0.84746 val_apk= 0.91002 time= 0.50647
Epoch: 0001 Iter: 1201 Edge: 0000 train_loss= 7.81395 val_roc= 0.88811 val_auprc= 0.87528 val_apk= 0.94187 time= 11.25271
Epoch: 0001 Iter: 1351 Edge: 0003 train_loss= 0.41312 val_roc= 0.82691 val_auprc= 0.81065 val_apk= 0.88838 time= 0.51723
Epoch: 0001 Iter: 1501 Edge: 0000 train_loss= 12.36913 val_roc= 0.89809 val_auprc= 0.88403 val_apk= 0.91002 time= 10.94971
Epoch: 0001 Iter: 1651 Edge: 0003 train_loss= 1.63497 val_roc= 0.81439 val_auprc= 0.81132 val_apk= 0.91002 time= 0.51430
Epoch: 0001 Iter: 1801 Edge: 0000 train_loss= 9.49334 val_roc= 0.89592 val_auprc= 0.88075 val_apk= 0.89125 time= 10.83780
Epoch: 0001 Iter: 1951 Edge: 0003 train_loss= 1.42752 val_roc= 0.83209 val_auprc= 0.83049 val_apk= 0.88241 time= 0.50736
Epoch: 0001 Iter: 2101 Edge: 0000 train_loss= 11.26401 val_roc= 0.89678 val_auprc= 0.88335 val_apk= 0.93902 time= 10.96761
Epoch: 0001 Iter: 2251 Edge: 0003 train_loss= 1.87466 val_roc= 0.85662 val_auprc= 0.84246 val_apk= 0.89410 time= 0.50839
Epoch: 0001 Iter: 2401 Edge: 0000 train_loss= 7.94036 val_roc= 0.90242 val_auprc= 0.88963 val_apk= 0.94860 time= 10.82699
Epoch: 0001 Iter: 2551 Edge: 0003 train_loss= 2.08428 val_roc= 0.86464 val_auprc= 0.85810 val_apk= 0.94039 time= 0.51751
Epoch: 0001 Iter: 2701 Edge: 0000 train_loss= 8.45070 val_roc= 0.90931 val_auprc= 0.89775 val_apk= 0.94187 time= 10.92162
Epoch: 0001 Iter: 2851 Edge: 0003 train_loss= 1.89403 val_roc= 0.83413 val_auprc= 0.82627 val_apk= 0.89656 time= 0.51273
Epoch: 0001 Iter: 3001 Edge: 0000 train_loss= 8.26208 val_roc= 0.89962 val_auprc= 0.88547 val_apk= 0.91002 time= 10.86429
Epoch: 0001 Iter: 3151 Edge: 0003 train_loss= 1.95335 val_roc= 0.85708 val_auprc= 0.85647 val_apk= 0.91573 time= 0.51226
Epoch: 0001 Iter: 3301 Edge: 0000 train_loss= 8.78674 val_roc= 0.90092 val_auprc= 0.88422 val_apk= 0.93394 time= 10.95959
    
    ------- result for epoch  1  -------
iter:  3372
Edge type= [00, 00, 00]
Edge type: 0000 Test AUROC score 0.90898
Edge type: 0000 Test AUPRC score 0.89518
Edge type: 0000 Test AP@k score 0.95041

Edge type= [00, 00, 01]
Edge type: 0001 Test AUROC score 0.92273
Edge type: 0001 Test AUPRC score 0.92172
Edge type: 0001 Test AP@k score 0.96554

Edge type= [00, 01, 00]
Edge type: 0002 Test AUROC score 0.86172
Edge type: 0002 Test AUPRC score 0.91847
Edge type: 0002 Test AP@k score 1.00000

Edge type= [01, 00, 00]
Edge type: 0003 Test AUROC score 0.82179
Edge type: 0003 Test AUPRC score 0.81464
Edge type: 0003 Test AP@k score 0.95208

Edge type= [01, 01, 00]
Edge type: 0004 Test AUROC score 0.09694
Edge type: 0004 Test AUPRC score 0.32623
Edge type: 0004 Test AP@k score 0.00876

Edge type= [01, 01, 01]
Edge type: 0005 Test AUROC score 0.79000
Edge type: 0005 Test AUPRC score 0.80784
Edge type: 0005 Test AP@k score 0.69262

Edge type= [01, 01, 02]
Edge type: 0006 Test AUROC score 0.26000
Edge type: 0006 Test AUPRC score 0.40142
Edge type: 0006 Test AP@k score 0.08935

Edge type= [01, 01, 03]
Edge type: 0007 Test AUROC score 0.32305
Edge type: 0007 Test AUPRC score 0.48684
Edge type: 0007 Test AP@k score 0.31592

Edge type= [01, 01, 04]
Edge type: 0008 Test AUROC score 0.60099
Edge type: 0008 Test AUPRC score 0.67202
Edge type: 0008 Test AP@k score 0.52932

Edge type= [01, 01, 05]
Edge type: 0009 Test AUROC score 0.38872
Edge type: 0009 Test AUPRC score 0.54113
Edge type: 0009 Test AP@k score 0.41498

Edge type= [01, 01, 06]
Edge type: 0010 Test AUROC score 0.38222
Edge type: 0010 Test AUPRC score 0.45173
Edge type: 0010 Test AP@k score 0.25730

Edge type= [01, 01, 07]
Edge type: 0011 Test AUROC score 0.55760
Edge type: 0011 Test AUPRC score 0.63753
Edge type: 0011 Test AP@k score 0.40877

Edge type= [01, 01, 08]
Edge type: 0012 Test AUROC score 0.62060
Edge type: 0012 Test AUPRC score 0.54144
Edge type: 0012 Test AP@k score 0.27744

Edge type= [01, 01, 09]
Edge type: 0013 Test AUROC score 0.79931
Edge type: 0013 Test AUPRC score 0.80835
Edge type: 0013 Test AP@k score 0.71766

Edge type= [01, 01, 10]
Edge type: 0014 Test AUROC score 0.78777
Edge type: 0014 Test AUPRC score 0.84306
Edge type: 0014 Test AP@k score 0.79763

Edge type= [01, 01, 11]
Edge type: 0015 Test AUROC score 0.66671
Edge type: 0015 Test AUPRC score 0.66418
Edge type: 0015 Test AP@k score 0.54500

==========================
Epoch: 0002 Iter: 0001 Edge: 0000 train_loss= 8.12220 val_roc= 0.90761 val_auprc= 0.89355 val_apk= 1.00000 time= 10.41194
Epoch: 0002 Iter: 0151 Edge: 0003 train_loss= 0.36273 val_roc= 0.84707 val_auprc= 0.85356 val_apk= 0.97295 time= 0.50771
Epoch: 0002 Iter: 0301 Edge: 0000 train_loss= 7.53806 val_roc= 0.90457 val_auprc= 0.88938 val_apk= 1.00000 time= 10.79806
Epoch: 0002 Iter: 0451 Edge: 0003 train_loss= 0.73338 val_roc= 0.79067 val_auprc= 0.79603 val_apk= 0.93568 time= 0.52655
Epoch: 0002 Iter: 0601 Edge: 0000 train_loss= 8.46834 val_roc= 0.90259 val_auprc= 0.88900 val_apk= 1.00000 time= 10.84521
Epoch: 0002 Iter: 0751 Edge: 0003 train_loss= 0.73632 val_roc= 0.81804 val_auprc= 0.81198 val_apk= 0.89371 time= 0.52891
Epoch: 0002 Iter: 0901 Edge: 0000 train_loss= 9.41491 val_roc= 0.91083 val_auprc= 0.89803 val_apk= 1.00000 time= 10.79598
Epoch: 0002 Iter: 1051 Edge: 0003 train_loss= 1.27700 val_roc= 0.82917 val_auprc= 0.83425 val_apk= 0.92002 time= 0.52063
Epoch: 0002 Iter: 1201 Edge: 0000 train_loss= 10.34900 val_roc= 0.90980 val_auprc= 0.89508 val_apk= 1.00000 time= 10.91745
Epoch: 0002 Iter: 1351 Edge: 0003 train_loss= 1.22544 val_roc= 0.84845 val_auprc= 0.85579 val_apk= 0.94187 time= 0.52388
Epoch: 0002 Iter: 1501 Edge: 0000 train_loss= 8.54812 val_roc= 0.91344 val_auprc= 0.89933 val_apk= 1.00000 time= 10.85292
Epoch: 0002 Iter: 1651 Edge: 0003 train_loss= 1.73409 val_roc= 0.84782 val_auprc= 0.85605 val_apk= 0.91598 time= 0.51540
Epoch: 0002 Iter: 1801 Edge: 0000 train_loss= 7.65239 val_roc= 0.90845 val_auprc= 0.89380 val_apk= 1.00000 time= 10.88423
Epoch: 0002 Iter: 1951 Edge: 0003 train_loss= 1.42032 val_roc= 0.84481 val_auprc= 0.85150 val_apk= 0.92920 time= 0.51501
Epoch: 0002 Iter: 2101 Edge: 0000 train_loss= 8.80838 val_roc= 0.91498 val_auprc= 0.90241 val_apk= 1.00000 time= 10.85421
Epoch: 0002 Iter: 2251 Edge: 0003 train_loss= 0.73358 val_roc= 0.82482 val_auprc= 0.82435 val_apk= 0.90461 time= 0.52314
Epoch: 0002 Iter: 2401 Edge: 0000 train_loss= 9.64343 val_roc= 0.91546 val_auprc= 0.90053 val_apk= 1.00000 time= 10.91251
Epoch: 0002 Iter: 2551 Edge: 0003 train_loss= 0.25617 val_roc= 0.83789 val_auprc= 0.83387 val_apk= 0.93950 time= 0.52532
Epoch: 0002 Iter: 2701 Edge: 0000 train_loss= 6.75686 val_roc= 0.91949 val_auprc= 0.90578 val_apk= 1.00000 time= 10.88229
Epoch: 0002 Iter: 2851 Edge: 0003 train_loss= 0.76044 val_roc= 0.82953 val_auprc= 0.81958 val_apk= 0.86650 time= 0.51214
Epoch: 0002 Iter: 3001 Edge: 0000 train_loss= 8.00954 val_roc= 0.91937 val_auprc= 0.90585 val_apk= 1.00000 time= 10.91594
Epoch: 0002 Iter: 3151 Edge: 0003 train_loss= 0.85795 val_roc= 0.85226 val_auprc= 0.84771 val_apk= 0.87451 time= 0.51234
Epoch: 0002 Iter: 3301 Edge: 0000 train_loss= 8.53289 val_roc= 0.91739 val_auprc= 0.90407 val_apk= 1.00000 time= 10.87467
Epoch: 0003 Iter: 0001 Edge: 0000 train_loss= 6.96217 val_roc= 0.91782 val_auprc= 0.90522 val_apk= 1.00000 time= 10.82358
Epoch: 0003 Iter: 0151 Edge: 0003 train_loss= 0.80439 val_roc= 0.84977 val_auprc= 0.84873 val_apk= 0.84557 time= 0.52812
Epoch: 0003 Iter: 0301 Edge: 0000 train_loss= 7.08641 val_roc= 0.92011 val_auprc= 0.90647 val_apk= 1.00000 time= 10.95092
Epoch: 0003 Iter: 0451 Edge: 0003 train_loss= 1.37928 val_roc= 0.86012 val_auprc= 0.85894 val_apk= 0.90986 time= 0.54731
Epoch: 0003 Iter: 0601 Edge: 0000 train_loss= 7.41495 val_roc= 0.92403 val_auprc= 0.91197 val_apk= 1.00000 time= 11.37579
Epoch: 0003 Iter: 0751 Edge: 0003 train_loss= 1.15039 val_roc= 0.84064 val_auprc= 0.83451 val_apk= 0.95335 time= 0.53111
Epoch: 0003 Iter: 0901 Edge: 0000 train_loss= 8.81705 val_roc= 0.92664 val_auprc= 0.91453 val_apk= 1.00000 time= 11.36662
Epoch: 0003 Iter: 1051 Edge: 0003 train_loss= 1.75312 val_roc= 0.83220 val_auprc= 0.82947 val_apk= 0.93128 time= 0.53893
Epoch: 0003 Iter: 1201 Edge: 0000 train_loss= 7.43565 val_roc= 0.93028 val_auprc= 0.91955 val_apk= 1.00000 time= 11.31834
Epoch: 0003 Iter: 1351 Edge: 0003 train_loss= 1.16652 val_roc= 0.84498 val_auprc= 0.84615 val_apk= 0.94187 time= 0.54750
Epoch: 0003 Iter: 1501 Edge: 0000 train_loss= 9.01468 val_roc= 0.92752 val_auprc= 0.91514 val_apk= 1.00000 time= 11.34140
Epoch: 0003 Iter: 1651 Edge: 0003 train_loss= 0.93617 val_roc= 0.83853 val_auprc= 0.83594 val_apk= 0.83634 time= 0.55046
Epoch: 0003 Iter: 1801 Edge: 0000 train_loss= 5.97988 val_roc= 0.93443 val_auprc= 0.92464 val_apk= 1.00000 time= 11.42516
Epoch: 0003 Iter: 1951 Edge: 0003 train_loss= 1.55376 val_roc= 0.84488 val_auprc= 0.84960 val_apk= 0.83956 time= 0.53847
Epoch: 0003 Iter: 2101 Edge: 0000 train_loss= 8.48238 val_roc= 0.93352 val_auprc= 0.92415 val_apk= 0.96292 time= 11.34888
Epoch: 0003 Iter: 2251 Edge: 0003 train_loss= 0.95945 val_roc= 0.84549 val_auprc= 0.86183 val_apk= 0.89853 time= 0.55243
Epoch: 0003 Iter: 2401 Edge: 0000 train_loss= 8.50496 val_roc= 0.93510 val_auprc= 0.92662 val_apk= 1.00000 time= 11.39349
Epoch: 0003 Iter: 2551 Edge: 0003 train_loss= 1.00291 val_roc= 0.81581 val_auprc= 0.83041 val_apk= 0.91002 time= 0.54732
Epoch: 0003 Iter: 2701 Edge: 0000 train_loss= 6.31295 val_roc= 0.93443 val_auprc= 0.92542 val_apk= 1.00000 time= 11.43348
Epoch: 0003 Iter: 2851 Edge: 0003 train_loss= 1.23473 val_roc= 0.85226 val_auprc= 0.86362 val_apk= 0.90125 time= 0.53919
Epoch: 0003 Iter: 3001 Edge: 0000 train_loss= 6.02399 val_roc= 0.93113 val_auprc= 0.91961 val_apk= 1.00000 time= 11.39838
Epoch: 0003 Iter: 3151 Edge: 0003 train_loss= 0.61855 val_roc= 0.87211 val_auprc= 0.86927 val_apk= 0.78440 time= 0.53582
Epoch: 0003 Iter: 3301 Edge: 0000 train_loss= 8.00246 val_roc= 0.93142 val_auprc= 0.91930 val_apk= 0.96992 time= 11.02481
Epoch: 0004 Iter: 0001 Edge: 0000 train_loss= 7.73337 val_roc= 0.93643 val_auprc= 0.92719 val_apk= 1.00000 time= 10.91872
Epoch: 0004 Iter: 0151 Edge: 0003 train_loss= 1.00287 val_roc= 0.82036 val_auprc= 0.82643 val_apk= 0.90871 time= 0.88535
Epoch: 0004 Iter: 0301 Edge: 0000 train_loss= 6.55671 val_roc= 0.93512 val_auprc= 0.92473 val_apk= 1.00000 time= 10.87946
Epoch: 0004 Iter: 0451 Edge: 0003 train_loss= 1.22448 val_roc= 0.86976 val_auprc= 0.89065 val_apk= 0.88183 time= 0.51441
Epoch: 0004 Iter: 0601 Edge: 0000 train_loss= 6.35205 val_roc= 0.93744 val_auprc= 0.92852 val_apk= 0.97119 time= 10.99504
Epoch: 0004 Iter: 0751 Edge: 0003 train_loss= 1.27772 val_roc= 0.84228 val_auprc= 0.84655 val_apk= 0.84942 time= 0.52093
Epoch: 0004 Iter: 0901 Edge: 0000 train_loss= 6.90123 val_roc= 0.93938 val_auprc= 0.93143 val_apk= 1.00000 time= 10.86528
Epoch: 0004 Iter: 1051 Edge: 0003 train_loss= 1.55072 val_roc= 0.83870 val_auprc= 0.84581 val_apk= 0.83216 time= 1.25516
Epoch: 0004 Iter: 1201 Edge: 0000 train_loss= 5.93197 val_roc= 0.93806 val_auprc= 0.92913 val_apk= 0.95638 time= 11.48473
Epoch: 0004 Iter: 1351 Edge: 0003 train_loss= 1.02980 val_roc= 0.85452 val_auprc= 0.86808 val_apk= 0.85167 time= 0.55143
Epoch: 0004 Iter: 1501 Edge: 0000 train_loss= 6.66161 val_roc= 0.93841 val_auprc= 0.92999 val_apk= 1.00000 time= 11.81464
Epoch: 0004 Iter: 1651 Edge: 0003 train_loss= 0.69044 val_roc= 0.84691 val_auprc= 0.86395 val_apk= 0.89880 time= 0.52021
Epoch: 0004 Iter: 1801 Edge: 0000 train_loss= 9.09931 val_roc= 0.93643 val_auprc= 0.92619 val_apk= 1.00000 time= 11.50896
Epoch: 0004 Iter: 1951 Edge: 0003 train_loss= 0.97285 val_roc= 0.79574 val_auprc= 0.80748 val_apk= 0.76035 time= 0.57063
Epoch: 0004 Iter: 2101 Edge: 0000 train_loss= 7.08561 val_roc= 0.93474 val_auprc= 0.92444 val_apk= 1.00000 time= 11.46039
Epoch: 0004 Iter: 2251 Edge: 0003 train_loss= 1.15497 val_roc= 0.88059 val_auprc= 0.88710 val_apk= 0.75750 time= 0.51927
