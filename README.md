# Clone, install things, try

```
git clone https://github.com/c0ntradicti0n/allennlp_vs_ampligraph.git
pip install -r requirements
./train_difference.sh simple_tagger_with_nyms.json
```
Logging will look different from allennlp, because of ampligraph logger settings.


# Simple Tagger

### Simple Tagger with nym embeddings

```
accuracy: 0.1321, accuracy3: 0.2620, loss: 5.1068 ||: 100%|##########| 125/125 [01:33<00:00,  1.33it/s]
accuracy: 0.1533, accuracy3: 0.2762, loss: 5.0046 ||: 100%|##########| 31/31 [00:11<00:00,  2.69it/s]
accuracy: 0.1711, accuracy3: 0.3622, loss: 4.8606 ||: 100%|##########| 125/125 [00:58<00:00,  2.15it/s][
accuracy: 0.1754, accuracy3: 0.3767, loss: 4.7747 ||: 100%|##########| 31/31 [00:02<00:00, 11.08it/s]
accuracy: 0.1952, accuracy3: 0.4015, loss: 4.6418 ||: 100%|##########| 125/125 [00:52<00:00,  2.37it/s][
accuracy: 0.1967, accuracy3: 0.5884, loss: 4.5643 ||: 100%|##########| 31/31 [00:02<00:00, 11.10it/s]
accuracy: 0.2337, accuracy3: 0.6093, loss: 4.4164 ||: 100%|##########| 125/125 [00:52<00:00,  2.37it/s]
accuracy: 0.2252, accuracy3: 0.6015, loss: 4.3673 ||: 100%|##########| 31/31 [00:02<00:00, 10.95it/s]
accuracy: 0.2495, accuracy3: 0.6180, loss: 4.2312 ||: 100%|##########| 125/125 [00:52<00:00,  2.37it/s]
accuracy: 0.2327, accuracy3: 0.6058, loss: 4.1946 ||: 100%|##########| 31/31 [00:02<00:00, 11.07it/s]
accuracy: 0.2577, accuracy3: 0.6224, loss: 4.0572 ||: 100%|##########| 125/125 [00:52<00:00,  2.40it/s]
accuracy: 0.2377, accuracy3: 0.6082, loss: 4.0366 ||: 100%|##########| 31/31 [00:02<00:00, 10.96it/s]
accuracy: 0.4146, accuracy3: 0.6249, loss: 3.9029 ||: 100%|##########| 125/125 [00:52<00:00,  2.39it/s]
accuracy: 0.4043, accuracy3: 0.6101, loss: 3.8956 ||: 100%|##########| 31/31 [00:02<00:00, 10.89it/s]
accuracy: 0.4332, accuracy3: 0.6281, loss: 3.7600 ||: 100%|##########| 125/125 [00:51<00:00,  2.42it/s]
accuracy: 0.4051, accuracy3: 0.6111, loss: 3.7585 ||: 100%|##########| 31/31 [00:02<00:00, 11.04it/s]
accuracy: 0.4385, accuracy3: 0.6310, loss: 3.6263 ||: 100%|##########| 125/125 [00:51<00:00,  2.42it/s]
accuracy: 0.4059, accuracy3: 0.6127, loss: 3.6343 ||: 100%|##########| 31/31 [00:02<00:00, 11.10it/s]
accuracy: 0.4408, accuracy3: 0.6303, loss: 3.5021 ||: 100%|##########| 125/125 [00:51<00:00,  2.42it/s]
accuracy: 0.4085, accuracy3: 0.6125, loss: 3.5183 ||: 100%|##########| 31/31 [00:02<00:00, 11.00it/s]
accuracy: 0.4428, accuracy3: 0.6320, loss: 3.3867 ||: 100%|##########| 125/125 [00:52<00:00,  2.39it/s]
accuracy: 0.4079, accuracy3: 0.6127, loss: 3.4125 ||: 100%|##########| 31/31 [00:02<00:00, 11.04it/s]
accuracy: 0.4431, accuracy3: 0.6315, loss: 3.2871 ||: 100%|##########| 125/125 [00:53<00:00,  2.35it/s]
accuracy: 0.4064, accuracy3: 0.6130, loss: 3.3170 ||: 100%|##########| 31/31 [00:02<00:00, 11.10it/s]
accuracy: 0.4442, accuracy3: 0.6323, loss: 3.1923 ||: 100%|##########| 125/125 [00:52<00:00,  2.39it/s]
accuracy: 0.4087, accuracy3: 0.6142, loss: 3.2241 ||: 100%|##########| 31/31 [00:02<00:00, 11.03it/s]
accuracy: 0.4453, accuracy3: 0.6327, loss: 3.1057 ||: 100%|##########| 125/125 [00:52<00:00,  2.38it/s]
accuracy: 0.4104, accuracy3: 0.6136, loss: 3.1446 ||: 100%|##########| 31/31 [00:02<00:00, 11.08it/s]
accuracy: 0.4485, accuracy3: 0.6338, loss: 3.0241 ||: 100%|##########| 125/125 [00:52<00:00,  2.40it/s]
accuracy: 0.4092, accuracy3: 0.6132, loss: 3.0732 ||: 100%|##########| 31/31 [00:02<00:00, 10.87it/s]
accuracy: 0.4491, accuracy3: 0.6331, loss: 2.9530 ||: 100%|##########| 125/125 [00:52<00:00,  2.38it/s]
accuracy: 0.4100, accuracy3: 0.6132, loss: 3.0045 ||: 100%|##########| 31/31 [00:02<00:00, 11.10it/s]
accuracy: 0.4504, accuracy3: 0.6328, loss: 2.8899 ||: 100%|##########| 125/125 [00:54<00:00,  2.31it/s]
accuracy: 0.4108, accuracy3: 0.6134, loss: 2.9449 ||: 100%|##########| 31/31 [00:02<00:00, 11.01it/s]
accuracy: 0.4491, accuracy3: 0.6333, loss: 2.8361 ||: 100%|##########| 125/125 [00:53<00:00,  2.36it/s]
accuracy: 0.4108, accuracy3: 0.6131, loss: 2.8896 ||: 100%|##########| 31/31 [00:02<00:00, 10.96it/s]
accuracy: 0.4495, accuracy3: 0.6325, loss: 2.7829 ||: 100%|##########| 125/125 [00:52<00:00,  2.38it/s]
accuracy: 0.4118, accuracy3: 0.6142, loss: 2.8406 ||: 100%|##########| 31/31 [00:02<00:00, 11.00it/s]
accuracy: 0.4524, accuracy3: 0.6346, loss: 2.7350 ||: 100%|##########| 125/125 [00:52<00:00,  2.37it/s]
accuracy: 0.4108, accuracy3: 0.6132, loss: 2.8015 ||: 100%|##########| 31/31 [00:02<00:00, 10.89it/s]
accuracy: 0.4517, accuracy3: 0.6332, loss: 2.6939 ||: 100%|##########| 125/125 [00:52<00:00,  2.37it/s]
accuracy: 0.4129, accuracy3: 0.6137, loss: 2.7599 ||: 100%|##########| 31/31 [00:02<00:00, 11.02it/s]
accuracy: 0.4527, accuracy3: 0.6347, loss: 2.6589 ||: 100%|##########| 125/125 [00:52<00:00,  2.36it/s]
accuracy: 0.4144, accuracy3: 0.6157, loss: 2.7216 ||: 100%|##########| 31/31 [00:02<00:00, 11.10it/s]
accuracy: 0.4518, accuracy3: 0.6335, loss: 2.6263 ||: 100%|##########| 125/125 [00:52<00:00,  2.37it/s]
accuracy: 0.4134, accuracy3: 0.6137, loss: 2.6955 ||: 100%|##########| 31/31 [00:02<00:00, 11.03it/s]
accuracy: 0.4531, accuracy3: 0.6345, loss: 2.5957 ||: 100%|##########| 125/125 [00:52<00:00,  2.38it/s]
accuracy: 0.4124, accuracy3: 0.6146, loss: 2.6703 ||: 100%|##########| 31/31 [00:02<00:00, 11.08it/s]
accuracy: 0.4534, accuracy3: 0.6346, loss: 2.5699 ||: 100%|##########| 125/125 [00:53<00:00,  2.33it/s]
accuracy: 0.4136, accuracy3: 0.6132, loss: 2.6453 ||: 100%|##########| 31/31 [00:02<00:00, 11.02it/s]
accuracy: 0.4547, accuracy3: 0.6350, loss: 2.5457 ||: 100%|##########| 125/125 [00:53<00:00,  2.35it/s]
accuracy: 0.4158, accuracy3: 0.6152, loss: 2.6213 ||: 100%|##########| 31/31 [00:02<00:00, 11.04it/s]
accuracy: 0.4514, accuracy3: 0.6332, loss: 2.5294 ||: 100%|##########| 125/125 [00:52<00:00,  2.39it/s]
accuracy: 0.4139, accuracy3: 0.6144, loss: 2.6029 ||: 100%|##########| 31/31 [00:02<00:00, 11.06it/s]
accuracy: 0.4524, accuracy3: 0.6341, loss: 2.5093 ||: 100%|##########| 125/125 [00:51<00:00,  2.40it/s]
accuracy: 0.4156, accuracy3: 0.6149, loss: 2.5814 ||: 100%|##########| 31/31 [00:02<00:00, 11.16it/s]
accuracy: 0.4516, accuracy3: 0.6325, loss: 2.4976 ||: 100%|##########| 125/125 [00:52<00:00,  2.36it/s]
accuracy: 0.4162, accuracy3: 0.6147, loss: 2.5704 ||: 100%|##########| 31/31 [00:02<00:00, 11.02it/s]
accuracy: 0.4508, accuracy3: 0.6324, loss: 2.4820 ||: 100%|##########| 125/125 [00:52<00:00,  2.37it/s]
accuracy: 0.4160, accuracy3: 0.6155, loss: 2.5575 ||: 100%|##########| 31/31 [00:02<00:00, 11.02it/s]
accuracy: 0.4540, accuracy3: 0.6345, loss: 2.4647 ||: 100%|##########| 125/125 [00:52<00:00,  2.39it/s]
accuracy: 0.4155, accuracy3: 0.6157, loss: 2.5463 ||: 100%|##########| 31/31 [00:02<00:00, 11.13it/s]
accuracy: 0.4528, accuracy3: 0.6338, loss: 2.4592 ||: 100%|##########| 125/125 [00:52<00:00,  2.38it/s]
accuracy: 0.4129, accuracy3: 0.6140, loss: 2.5419 ||: 100%|##########| 31/31 [00:02<00:00, 11.13it/s]
accuracy: 0.4537, accuracy3: 0.6351, loss: 2.4457 ||: 100%|##########| 125/125 [00:52<00:00,  2.36it/s]
accuracy: 0.4149, accuracy3: 0.6160, loss: 2.5261 ||: 100%|##########| 31/31 [00:02<00:00, 11.03it/s]
accuracy: 0.4545, accuracy3: 0.6352, loss: 2.4373 ||: 100%|##########| 125/125 [00:52<00:00,  2.36it/s]
accuracy: 0.4168, accuracy3: 0.6141, loss: 2.5208 ||: 100%|##########| 31/31 [00:02<00:00, 11.01it/s]
accuracy: 0.4528, accuracy3: 0.6339, loss: 2.4320 ||: 100%|##########| 125/125 [00:52<00:00,  2.39it/s]
accuracy: 0.4131, accuracy3: 0.6140, loss: 2.5164 ||: 100%|##########| 31/31 [00:02<00:00, 11.05it/s]
accuracy: 0.4540, accuracy3: 0.6350, loss: 2.4229 ||: 100%|##########| 125/125 [00:52<00:00,  2.38it/s]
accuracy: 0.4179, accuracy3: 0.6163, loss: 2.5031 ||: 100%|##########| 31/31 [00:02<00:00, 11.07it/s]
accuracy: 0.4514, accuracy3: 0.6324, loss: 2.4214 ||: 100%|##########| 125/125 [00:52<00:00,  2.38it/s]
accuracy: 0.4177, accuracy3: 0.6162, loss: 2.4965 ||: 100%|##########| 31/31 [00:02<00:00, 10.88it/s]
accuracy: 0.4523, accuracy3: 0.6331, loss: 2.4154 ||: 100%|##########| 125/125 [00:52<00:00,  2.37it/s]
accuracy: 0.4167, accuracy3: 0.6152, loss: 2.4939 ||: 100%|##########| 31/31 [00:02<00:00, 11.04it/s]
accuracy: 0.4519, accuracy3: 0.6328, loss: 2.4135 ||: 100%|##########| 125/125 [00:53<00:00,  2.34it/s]
accuracy: 0.4171, accuracy3: 0.6155, loss: 2.4890 ||: 100%|##########| 31/31 [00:02<00:00, 11.10it/s]
accuracy: 0.4524, accuracy3: 0.6330, loss: 2.4094 ||: 100%|##########| 125/125 [00:53<00:00,  2.34it/s]
accuracy: 0.4173, accuracy3: 0.6157, loss: 2.4868 ||: 100%|##########| 31/31 [00:02<00:00, 10.78it/s]
```

Seems to be stupid with knowledge, because it refuses to learn below some loss. Without knowledge training looks like:
### Simple Tagger without nym embeddings
```
./train_difference.sh simple_tagger_without_nyms.json  
```
works fine. Learns epoch for epoch more and more
```
2019-10-15 12:45:23,697 - INFO - allennlp.training.trainer - Beginning training.
2019-10-15 12:45:23,697 - INFO - allennlp.training.trainer - Epoch 0/39
2019-10-15 12:45:23,697 - INFO - allennlp.training.trainer - Peak CPU memory usage MB: 1919.684
2019-10-15 12:45:23,739 - INFO - allennlp.training.trainer - Training
accuracy: 0.1656, accuracy3: 0.3559, loss: 3.5716 ||: 100%|##########| 125/125 [00:37<00:00,  3.32it/s]
2019-10-15 12:46:01,372 - INFO - allennlp.training.trainer - Validating
accuracy: 0.1852, accuracy3: 0.3841, loss: 3.2306 ||: 100%|##########| 31/31 [00:02<00:00, 14.47it/s]
2019-10-15 12:46:03,515 - INFO - allennlp.training.tensorboard_writer -                   Training |  Validation
2019-10-15 12:46:03,515 - INFO - allennlp.training.tensorboard_writer - cpu_memory_MB |  1919.684  |       N/A
2019-10-15 12:46:03,516 - INFO - allennlp.training.tensorboard_writer - accuracy3     |     0.356  |     0.384
2019-10-15 12:46:03,516 - INFO - allennlp.training.tensorboard_writer - accuracy      |     0.166  |     0.185
2019-10-15 12:46:03,517 - INFO - allennlp.training.tensorboard_writer - loss          |     3.572  |     3.231
2019-10-15 12:46:03,540 - INFO - allennlp.training.checkpointer - Best validation performance so far. Copying weights to './output/simple_tagger_without_nyms.json//best.th'.
2019-10-15 12:46:03,546 - INFO - allennlp.training.trainer - Epoch duration: 0:00:39.848754
2019-10-15 12:46:03,548 - INFO - allennlp.training.trainer - Estimated training time remaining: 0:25:54
2019-10-15 12:46:03,548 - INFO - allennlp.training.trainer - Epoch 1/39
2019-10-15 12:46:03,548 - INFO - allennlp.training.trainer - Peak CPU memory usage MB: 2232.72
2019-10-15 12:46:03,602 - INFO - allennlp.training.trainer - Training
accuracy: 0.2511, accuracy3: 0.4494, loss: 3.0232 ||: 100%|##########| 125/125 [00:38<00:00,  3.23it/s]
2019-10-15 12:46:42,335 - INFO - allennlp.training.trainer - Validating
accuracy: 0.3684, accuracy3: 0.5976, loss: 2.4891 ||: 100%|##########| 31/31 [00:01<00:00, 20.92it/s]
2019-10-15 12:46:43,818 - INFO - allennlp.training.tensorboard_writer -                   Training |  Validation
2019-10-15 12:46:43,818 - INFO - allennlp.training.tensorboard_writer - cpu_memory_MB |  2232.720  |       N/A
2019-10-15 12:46:43,819 - INFO - allennlp.training.tensorboard_writer - accuracy3     |     0.449  |     0.598
2019-10-15 12:46:43,819 - INFO - allennlp.training.tensorboard_writer - accuracy      |     0.251  |     0.368
2019-10-15 12:46:43,820 - INFO - allennlp.training.tensorboard_writer - loss          |     3.023  |     2.489
2019-10-15 12:46:43,841 - INFO - allennlp.training.checkpointer - Best validation performance so far. Copying weights to './output/simple_tagger_without_nyms.json//best.th'.
2019-10-15 12:46:43,850 - INFO - allennlp.training.trainer - Epoch duration: 0:00:40.301615
2019-10-15 12:46:43,850 - INFO - allennlp.training.trainer - Estimated training time remaining: 0:25:22
2019-10-15 12:46:43,850 - INFO - allennlp.training.trainer - Epoch 2/39
2019-10-15 12:46:43,850 - INFO - allennlp.training.trainer - Peak CPU memory usage MB: 2324.584
2019-10-15 12:46:43,918 - INFO - allennlp.training.trainer - Training
accuracy: 0.4859, accuracy3: 0.6952, loss: 2.0343 ||: 100%|##########| 125/125 [00:43<00:00,  2.91it/s]
2019-10-15 12:47:26,928 - INFO - allennlp.training.trainer - Validating
accuracy: 0.5964, accuracy3: 0.7851, loss: 1.6124 ||: 100%|##########| 31/31 [00:01<00:00, 18.35it/s]
2019-10-15 12:47:28,618 - INFO - allennlp.training.tensorboard_writer -                   Training |  Validation
2019-10-15 12:47:28,618 - INFO - allennlp.training.tensorboard_writer - cpu_memory_MB |  2324.584  |       N/A
2019-10-15 12:47:28,619 - INFO - allennlp.training.tensorboard_writer - accuracy3     |     0.695  |     0.785
2019-10-15 12:47:28,619 - INFO - allennlp.training.tensorboard_writer - accuracy      |     0.486  |     0.596
2019-10-15 12:47:28,620 - INFO - allennlp.training.tensorboard_writer - loss          |     2.034  |     1.612
2019-10-15 12:47:28,637 - INFO - allennlp.training.checkpointer - Best validation performance so far. Copying weights to './output/simple_tagger_without_nyms.json//best.th'.
2019-10-15 12:47:28,646 - INFO - allennlp.training.trainer - Epoch duration: 0:00:44.796049
2019-10-15 12:47:28,646 - INFO - allennlp.training.trainer - Estimated training time remaining: 0:25:41
2019-10-15 12:47:28,646 - INFO - allennlp.training.trainer - Epoch 3/39
2019-10-15 12:47:28,646 - INFO - allennlp.training.trainer - Peak CPU memory usage MB: 2370.756
2019-10-15 12:47:28,712 - INFO - allennlp.training.trainer - Training
accuracy: 0.6571, accuracy3: 0.8196, loss: 1.3888 ||: 100%|##########| 125/125 [00:40<00:00,  3.08it/s]
2019-10-15 12:48:09,280 - INFO - allennlp.training.trainer - Validating
accuracy: 0.7213, accuracy3: 0.8532, loss: 1.1983 ||: 100%|##########| 31/31 [00:01<00:00, 19.22it/s]
2019-10-15 12:48:10,894 - INFO - allennlp.training.tensorboard_writer -                   Training |  Validation
2019-10-15 12:48:10,895 - INFO - allennlp.training.tensorboard_writer - cpu_memory_MB |  2370.756  |       N/A
2019-10-15 12:48:10,895 - INFO - allennlp.training.tensorboard_writer - accuracy3     |     0.820  |     0.853
2019-10-15 12:48:10,895 - INFO - allennlp.training.tensorboard_writer - accuracy      |     0.657  |     0.721
2019-10-15 12:48:10,896 - INFO - allennlp.training.tensorboard_writer - loss          |     1.389  |     1.198
2019-10-15 12:48:10,917 - INFO - allennlp.training.checkpointer - Best validation performance so far. Copying weights to './output/simple_tagger_without_nyms.json//best.th'.
2019-10-15 12:48:10,925 - INFO - allennlp.training.trainer - Epoch duration: 0:00:42.278537
2019-10-15 12:48:10,925 - INFO - allennlp.training.trainer - Estimated training time remaining: 0:25:05
2019-10-15 12:48:10,925 - INFO - allennlp.training.trainer - Epoch 4/39
2019-10-15 12:48:10,925 - INFO - allennlp.training.trainer - Peak CPU memory usage MB: 2420.084
2019-10-15 12:48:11,003 - INFO - allennlp.training.trainer - Training
accuracy: 0.7611, accuracy3: 0.8917, loss: 1.0077 ||: 100%|##########| 125/125 [00:42<00:00,  2.96it/s]
2019-10-15 12:48:53,291 - INFO - allennlp.training.trainer - Validating
accuracy: 0.7871, accuracy3: 0.8971, loss: 0.9419 ||: 100%|##########| 31/31 [00:01<00:00, 17.89it/s]
2019-10-15 12:48:55,025 - INFO - allennlp.training.tensorboard_writer -                   Training |  Validation
2019-10-15 12:48:55,025 - INFO - allennlp.training.tensorboard_writer - cpu_memory_MB |  2420.084  |       N/A
2019-10-15 12:48:55,025 - INFO - allennlp.training.tensorboard_writer - accuracy3     |     0.892  |     0.897
2019-10-15 12:48:55,026 - INFO - allennlp.training.tensorboard_writer - accuracy      |     0.761  |     0.787
2019-10-15 12:48:55,026 - INFO - allennlp.training.tensorboard_writer - loss          |     1.008  |     0.942
2019-10-15 12:48:55,043 - INFO - allennlp.training.checkpointer - Best validation performance so far. Copying weights to './output/simple_tagger_without_nyms.json//best.th'.
2019-10-15 12:48:55,050 - INFO - allennlp.training.trainer - Epoch duration: 0:00:44.124698
2019-10-15 12:48:55,050 - INFO - allennlp.training.trainer - Estimated training time remaining: 0:24:39
2019-10-15 12:48:55,050 - INFO - allennlp.training.trainer - Epoch 5/39
2019-10-15 12:48:55,050 - INFO - allennlp.training.trainer - Peak CPU memory usage MB: 2470.244
2019-10-15 12:48:55,117 - INFO - allennlp.training.trainer - Training
accuracy: 0.8307, accuracy3: 0.9313, loss: 0.7537 ||: 100%|##########| 125/125 [00:39<00:00,  3.17it/s]
2019-10-15 12:49:34,615 - INFO - allennlp.training.trainer - Validating
accuracy: 0.8269, accuracy3: 0.9214, loss: 0.7693 ||: 100%|##########| 31/31 [00:01<00:00, 20.66it/s]
2019-10-15 12:49:36,117 - INFO - allennlp.training.tensorboard_writer -                   Training |  Validation
2019-10-15 12:49:36,118 - INFO - allennlp.training.tensorboard_writer - cpu_memory_MB |  2470.244  |       N/A
2019-10-15 12:49:36,118 - INFO - allennlp.training.tensorboard_writer - accuracy3     |     0.931  |     0.921
2019-10-15 12:49:36,119 - INFO - allennlp.training.tensorboard_writer - accuracy      |     0.831  |     0.827
2019-10-15 12:49:36,119 - INFO - allennlp.training.tensorboard_writer - loss          |     0.754  |     0.769
2019-10-15 12:49:36,139 - INFO - allennlp.training.checkpointer - Best validation performance so far. Copying weights to './output/simple_tagger_without_nyms.json//best.th'.

```
# Span tagging with Conditional Random Field
### With nym embeddings
```./train_difference.sh simple_span_tagger_without.json```

either gives a math error:

```
        last_tags = tags.gather(0, last_tag_index.view(1, batch_size)).squeeze(0)
    RuntimeError: Invalid index in gather at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:657
     0%|          | 0/439 [00:01<?, ?it/s]

```

Or loss runs to negative value and ends with a predicted tag out, that was not given in the training set, in other cases.

```
accuracy: 0.1889, accuracy3: 0.1889, precision-overall: 0.0229, recall-overall: 0.0015, f1-measure-overall: 0.0027, loss: 2212.1269 ||:  51%|#####1    | 19/37 [04:04<03
accuracy: 0.1892, accuracy3: 0.1892, precision-overall: 0.0455, recall-overall: 0.0036, f1-measure-overall: 0.0066, loss: 2216.5326 ||:  54%|#####4    | 20/37 [04:17<03
accuracy: 0.1891, accuracy3: 0.1891, precision-overall: 0.0632, recall-overall: 0.0063, f1-measure-overall: 0.0114, loss: 2219.2800 ||:  57%|#####6    | 21/37 [04:29<03
accuracy: 0.1896, accuracy3: 0.1896, precision-overall: 0.0865, recall-overall: 0.0090, f1-measure-overall: 0.0162, loss: 2214.7909 ||:  59%|#####9    | 22/37 [04:42<03
accuracy: 0.1893, accuracy3: 0.1893, precision-overall: 0.0865, recall-overall: 0.0086, f1-measure-overall: 0.0156, loss: 2209.9617 ||:  62%|######2   | 23/37 [04:54<02
accuracy: 0.1877, accuracy3: 0.1877, precision-overall: 0.0865, recall-overall: 0.0082, f1-measure-overall: 0.0150, loss: 2209.3227 ||:  65%|######4   | 24/37 [05:06<02
accuracy: 0.1873, accuracy3: 0.1873, precision-overall: 0.0969, recall-overall: 0.0103, f1-measure-overall: 0.0186, loss: 2202.3855 ||:  68%|######7   | 25/37 [05:19<02
accuracy: 0.1883, accuracy3: 0.1883, precision-overall: 0.1014, recall-overall: 0.0122, f1-measure-overall: 0.0217, loss: 2201.3712 ||:  70%|#######   | 26/37 [05:31<02
accuracy: 0.1912, accuracy3: 0.1912, precision-overall: 0.1165, recall-overall: 0.0156, f1-measure-overall: 0.0276, loss: 2192.0540 ||:  73%|#######2  | 27/37 [05:42<01
accuracy: 0.1915, accuracy3: 0.1915, precision-overall: 0.1206, recall-overall: 0.0160, f1-measure-overall: 0.0283, loss: 2173.3288 ||:  76%|#######5  | 28/37 [05:55<01
accuracy: 0.1915, accuracy3: 0.1915, precision-overall: 0.1195, recall-overall: 0.0159, f1-measure-overall: 0.0280, loss: 2155.9421 ||:  78%|#######8  | 29/37 [06:08<01
accuracy: 0.1918, accuracy3: 0.1918, precision-overall: 0.1192, recall-overall: 0.0189, f1-measure-overall: 0.0326, loss: 2121.5115 ||:  81%|########1 | 30/37 [06:19<01
accuracy: 0.1928, accuracy3: 0.1928, precision-overall: 0.1187, recall-overall: 0.0195, f1-measure-overall: 0.0335, loss: 2074.0345 ||:  84%|########3 | 31/37 [06:31<01
accuracy: 0.1915, accuracy3: 0.1915, precision-overall: 0.1178, recall-overall: 0.0189, f1-measure-overall: 0.0326, loss: 1950.1953 ||:  86%|########6 | 32/37 [06:43<01
accuracy: 0.1905, accuracy3: 0.1905, precision-overall: 0.1178, recall-overall: 0.0183, f1-measure-overall: 0.0317, loss: 1692.1747 ||:  89%|########9 | 33/37 [06:58<00
accuracy: 0.1905, accuracy3: 0.1905, precision-overall: 0.1114, recall-overall: 0.0179, f1-measure-overall: 0.0308, loss: 926.1686 ||:  92%|#########1| 34/37 [07:11<00:
accuracy: 0.1903, accuracy3: 0.1903, precision-overall: 0.0947, recall-overall: 0.0201, f1-measure-overall: 0.0331, loss: 51.6322 ||:  95%|#########4| 35/37 [07:23<00:2
accuracy: 0.1905, accuracy3: 0.1905, precision-overall: 0.0947, recall-overall: 0.0195, f1-measure-overall: 0.0323, loss: -4033.6692 ||:  97%|#########7| 36/37 [07:36<00:12, 12.65s/it]
Traceback (most recent call last):
  File "/roedel/home/finn/ai-difference/venv/bin/allennlp", line 11, in <module>
    sys.exit(run())
  File "/roedel/home/finn/ai-difference/venv/lib/python3.7/site-packages/allennlp/run.py", line 18, in run
    main(prog="allennlp")
  File "/roedel/home/finn/ai-difference/venv/lib/python3.7/site-packages/allennlp/commands/__init__.py", line 102, in main
    args.func(args)
  File "/roedel/home/finn/ai-difference/venv/lib/python3.7/site-packages/allennlp/commands/train.py", line 124, in train_model_from_args
    args.cache_prefix)
  File "/roedel/home/finn/ai-difference/venv/lib/python3.7/site-packages/allennlp/commands/train.py", line 168, in train_model_from_file
    cache_directory, cache_prefix)
  File "/roedel/home/finn/ai-difference/venv/lib/python3.7/site-packages/allennlp/commands/train.py", line 252, in train_model
    metrics = trainer.train()
  File "/roedel/home/finn/ai-difference/venv/lib/python3.7/site-packages/allennlp/training/trainer.py", line 478, in train
    train_metrics = self._train_epoch(epoch)
  File "/roedel/home/finn/ai-difference/venv/lib/python3.7/site-packages/allennlp/training/trainer.py", line 320, in _train_epoch
    loss = self.batch_loss(batch_group, for_training=True)
  File "/roedel/home/finn/ai-difference/venv/lib/python3.7/site-packages/allennlp/training/trainer.py", line 261, in batch_loss
    output_dict = self.model(**batch)
  File "/roedel/home/finn/ai-difference/venv/lib/python3.7/site-packages/torch/nn/modules/module.py", line 547, in __call__
    result = self.forward(*input, **kwargs)
  File "/roedel/home/finn/ai-difference/venv/lib/python3.7/site-packages/allennlp/models/crf_tagger.py", line 217, in forward
    class_probabilities[i, j, tag_id] = 1
IndexError: index 20 is out of bounds for dimension 2 with size 20



```

### Whithout nym embeddings
``` ./train_difference.sh span_tagger_with_nyms.json ```

works fine...

```
2019-10-15 12:40:55,864 - INFO - allennlp.training.trainer - Beginning training.
2019-10-15 12:40:55,865 - INFO - allennlp.training.trainer - Epoch 0/39
2019-10-15 12:40:55,865 - INFO - allennlp.training.trainer - Peak CPU memory usage MB: 1967.748
2019-10-15 12:40:55,905 - INFO - allennlp.training.trainer - Training
accuracy: 0.8366, accuracy3: 0.8453, precision-overall: 0.2904, recall-overall: 0.1447, f1-measure-overall: 0.1931, loss: 330.0508 ||:  83%|########3 | 365/439 [01:25<00:16,  4.42it/s]^
...
...
...

```